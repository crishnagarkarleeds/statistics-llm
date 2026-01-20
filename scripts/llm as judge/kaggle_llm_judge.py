# ===== LLM-AS-JUDGE EVALUATION =====

# Install required packages
!pip install transformers accelerate bitsandbytes torch -q
!pip install sentence-transformers bert-score nltk -q

import json
import pandas as pd
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

class KaggleLLMJudge:
    def __init__(self):
        """LLM Judge optimized for Kaggle environment"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Using device: {self.device}")
        
        # File paths - update these to match your Kaggle datasets
        self.response_files = {
            'DeepSeek_Base': '/kaggle/input/finalllms/final jsons/DeepSeek r1 responses.json',
            'LLaMA_Base': '/kaggle/input/finalllama/llama3_responses_fixed.json',
            'LLaMA_FT': '/kaggle/input/finalllms/final jsons/Fine tuned LLAMA.json',
            'Mistral_Base': '/kaggle/input/mistralbase/Mistral copy.json',
            'Mistral_FT': '/kaggle/input/finalllms/final jsons/Fine tuned Mistral.json'
        }
        
        self.human_csv_path = '/kaggle/input/scores/Muxi Li completed markheet(Sheet1).csv'
        self.solution_bank_path = '/kaggle/input/finalllms/final jsons/solution bank.json'
        
        # Evaluation prompt
        self.judge_prompt = """Rate this response on three aspects (0-5 scale, allow decimals like 3.5):

**Problem:** {problem}
**Correct Answer:** {correct_solution}  
**Student Response:** {student_response}

**Rate on:**
1. **Correctness** - Is the final answer right?
2. **Explanation** - Are steps clear and logical?  
3. **Reasoning** - Shows good understanding?

**Respond with exactly 3 numbers separated by commas:**
Correctness, Explanation, Reasoning

**Example format:** 4.5, 3.0, 4.0"""

    def load_judge_model(self, model_name):
        """Load judge model with memory optimization"""
        print(f"🤖 Loading judge model: {model_name}")
        
        # Memory-efficient configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded successfully")
        return model, tokenizer

    def get_judge_scores(self, model, tokenizer, problem, correct_solution, student_response):
        """Get scores from judge model"""
        try:
            # Format prompt
            prompt = self.judge_prompt.format(
                problem=problem[:500],  # Truncate long problems
                correct_solution=correct_solution[:300],
                student_response=student_response[:500]
            )
            
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors="pt", 
                                    truncation=True, max_length=1500)
            inputs = inputs.to(self.device)
            
            # Generate with multiple attempts
            for attempt in range(3):  # Try 3 times
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=50,
                        min_new_tokens=5,
                        do_sample=True,
                        temperature=0.7 + (attempt * 0.1),  # Increase temp each attempt
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                
                # Try to parse scores
                scores = self.parse_scores(response)
                if scores != (2.5, 2.5, 2.5):  # If not default fallback
                    return scores
            
            # If all attempts failed, return defaults
            return (2.5, 2.5, 2.5)
            
        except Exception as e:
            print(f"⚠️ Error in scoring: {e}")
            return (2.5, 2.5, 2.5)

    def parse_scores(self, response):
        """Extract three scores from model response"""
        try:
            # Look for comma-separated numbers
            pattern = r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)'
            match = re.search(pattern, response)
            
            if match:
                scores = [float(match.group(i)) for i in [1, 2, 3]]
                # Clamp to 0-5 range
                scores = [max(0, min(5, score)) for score in scores]
                return tuple(scores)
            
            # Fallback: find any three numbers
            numbers = re.findall(r'\d+\.?\d*', response)
            if len(numbers) >= 3:
                scores = [float(n) for n in numbers[:3]]
                scores = [max(0, min(5, score)) for score in scores]
                return tuple(scores)
                
        except:
            pass
        
        return (2.5, 2.5, 2.5)  # Default fallback

    def load_solution_bank(self):
        """Load reference solutions"""
        print("📚 Loading solution bank...")
        try:
            with open(self.solution_bank_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            solutions = []
            questions = data if isinstance(data, list) else data.get('questions', [])
            
            for q in questions:
                if isinstance(q, dict):
                    # Try different solution fields
                    solution = (q.get('conclusion') or 
                              q.get('solution') or 
                              q.get('answer') or
                              ' '.join(str(s) for s in q.get('solution_steps', [])) or
                              'No solution available')
                    solutions.append(str(solution))
            
            print(f"✅ Loaded {len(solutions)} reference solutions")
            return solutions
            
        except Exception as e:
            print(f"❌ Error loading solutions: {e}")
            return []

    def extract_model_data(self, file_path, model_name):
        """Extract problems and responses from model JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problems = []
            responses = []
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Extract problem
                problem = item.get('problem', 'No problem found')
                problems.append(str(problem))
                
                # Extract response based on model type
                if model_name == 'DeepSeek_Base':
                    mr = item.get('model_response', {})
                    response = mr.get('response', 'No response') if isinstance(mr, dict) else str(mr)
                
                elif model_name == 'LLaMA_Base':
                    mr = item.get('model_response', {})
                    if isinstance(mr, dict):
                        response = mr.get('solution', mr.get('response', 'No response'))
                    else:
                        response = str(mr) if mr else 'No response'
                
                elif model_name == 'LLaMA_FT':
                    response = item.get('llama_answer', 'No response')
                
                elif model_name == 'Mistral_Base':
                    mr = item.get('model_response', {})
                    response = mr.get('solution', 'No response') if isinstance(mr, dict) else str(mr)
                
                elif model_name == 'Mistral_FT':
                    response = item.get('aire_mistral_answer', 'No response')
                
                else:
                    response = 'Unknown model type'
                
                responses.append(str(response))
            
            print(f"✅ Extracted {len(responses)} responses from {model_name}")
            return problems, responses
            
        except Exception as e:
            print(f"❌ Error extracting {model_name}: {e}")
            return [], []

    def evaluate_single_model_responses(self, judge_model_name, model_name, file_path):
        """Evaluate one model's responses with one judge"""
        print(f"\n🔥 Evaluating {model_name} with {judge_model_name}")
        print("="*60)
        
        # Load data
        solutions = self.load_solution_bank()
        problems, responses = self.extract_model_data(file_path, model_name)
        
        if not problems or not responses or not solutions:
            print("❌ Failed to load required data")
            return None
        
        # Take minimum length
        min_len = min(len(problems), len(responses), len(solutions), 50)  # Max 50
        problems = problems[:min_len]
        responses = responses[:min_len]
        solutions = solutions[:min_len]
        
        print(f"📊 Evaluating {min_len} question-response pairs")
        
        # Load judge model
        judge_model, judge_tokenizer = self.load_judge_model(judge_model_name)
        
        # Evaluate each response
        results = []
        for i in tqdm(range(min_len), desc=f"Evaluating {model_name}"):
            correctness, explanation, reasoning = self.get_judge_scores(
                judge_model, judge_tokenizer, 
                problems[i], solutions[i], responses[i]
            )
            
            # Calculate weighted score (40% + 35% + 25%)
            weighted = (correctness * 0.40) + (explanation * 0.35) + (reasoning * 0.25)
            
            results.append({
                'question_id': i + 1,
                'target_model': model_name,
                'judge_model': judge_model_name.split('/')[-1],
                'problem': problems[i][:200] + "..." if len(problems[i]) > 200 else problems[i],
                'student_response': responses[i][:200] + "..." if len(responses[i]) > 200 else responses[i],
                'correctness': round(correctness, 2),
                'explanation': round(explanation, 2),
                'reasoning': round(reasoning, 2),
                'weighted': round(weighted, 2)
            })
        
        # Clean up memory
        del judge_model, judge_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Print summary
        print(f"\n📊 {model_name} Results Summary:")
        print(f"   Correctness: {df['correctness'].mean():.2f} ± {df['correctness'].std():.2f}")
        print(f"   Explanation: {df['explanation'].mean():.2f} ± {df['explanation'].std():.2f}")
        print(f"   Reasoning:   {df['reasoning'].mean():.2f} ± {df['reasoning'].std():.2f}")
        print(f"   Weighted:    {df['weighted'].mean():.2f} ± {df['weighted'].std():.2f}")
        
        return df

    def run_all_evaluations(self, judge_models):
        """Run evaluations for all models with all judges"""
        print("🚀 STARTING COMPREHENSIVE LLM-AS-JUDGE EVALUATION")
        print("="*80)
        
        all_results = []
        
        for judge_name in judge_models:
            print(f"\n🤖 Using Judge: {judge_name}")
            
            for model_name, file_path in self.response_files.items():
                try:
                    df = self.evaluate_single_model_responses(judge_name, model_name, file_path)
                    if df is not None:
                        all_results.append(df)
                        
                        # Save individual results
                        filename = f"judge_{judge_name.split('/')[-1]}_{model_name}_scores.csv"
                        df.to_csv(filename, index=False)
                        print(f"💾 Saved: {filename}")
                        
                except Exception as e:
                    print(f"❌ Failed to evaluate {model_name} with {judge_name}: {e}")
                    continue
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df.to_csv('all_llm_judge_results.csv', index=False)
            print(f"\n✅ EVALUATION COMPLETE!")
            print(f"📁 Combined results saved: all_llm_judge_results.csv")
            print(f"📊 Total evaluations: {len(combined_df)}")
            
            # Generate summary table
            summary = combined_df.groupby(['judge_model', 'target_model'])['weighted'].agg(['mean', 'std']).round(3)
            print(f"\n📈 SUMMARY TABLE:")
            print(summary)
            
            return combined_df
        else:
            print("❌ No successful evaluations completed")
            return None

# ===== EXECUTION SECTION =====

def main():
    """Main execution function"""
    
    # Initialize evaluator
    evaluator = KaggleLLMJudge()
    
    # Define judge models to use
    judge_models = [
        "mistralai/Mistral-Nemo-Instruct-2407",
         "meta-llama/Meta-Llama-3-8B-Instruct",
         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]
    
    print("🎯 JUDGE MODELS TO USE:")
    for i, model in enumerate(judge_models, 1):
        print(f"  {i}. {model}")
    
    # Run evaluations
    results_df = evaluator.run_all_evaluations(judge_models)
    
    if results_df is not None:
        print("\n🎉 SUCCESS! Check the output files:")
        print("   - all_llm_judge_results.csv (combined results)")
        print("   - judge_*_scores.csv (individual judge results)")
        
        return results_df
    else:
        print("\n Evaluation failed. Check error messages above.")
        return None

# Run the evaluation
if __name__ == "__main__":
    final_results = main()
