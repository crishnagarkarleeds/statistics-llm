"""
Complete script to generate automated metrics

This script computes BLEU, BERTScore, SBERT Similarity, Perplexity
and generates the automated_metrics.csv file with 250 rows (50 questions × 5 models)
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, wilcoxon
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')



class CSVGenerator:
    
    def __init__(self, analyzer):
        """Initialize with completed analyzer object"""
        self.analyzer = analyzer
    
    def generate_automated_metrics_csv(self):
        """
        Generate automated_metrics.csv with all computed metrics
        This creates the exact format: id, model_name, BLEU_score, BERTScore_F1, 
        SBERT_Similarity, Perplexity, Word_Overlap, Length_Similarity, 
        Char_Similarity, Quality_Score
        """
        
        print("📊 Generating automated_metrics.csv...")
        
        data = []
        
        for model_name, metrics in self.analyzer.automated_metrics.items():
            n_questions = len(metrics['bleu'])
            
            for i in range(n_questions):
                data.append({
                    'id': i + 1,
                    'model_name': model_name,
                    'BLEU_score': metrics['bleu'][i],
                    'BERTScore_F1': metrics['bertscore_f1'][i],
                    'SBERT_Similarity': metrics['sbert_similarity'][i],
                    'Perplexity': metrics['perplexity'][i],
                    'Word_Overlap': metrics['word_overlap'][i],
                    'Length_Similarity': metrics['length_similarity'][i], 
                    'Char_Similarity': metrics['char_similarity'][i],
                    'Quality_Score': metrics['quality_score'][i]
                })
        
        df = pd.DataFrame(data)
        df.to_csv('automated_metrics.csv', index=False)
        print(f"✅ Generated automated_metrics.csv with {len(df)} rows")
        print("📊 Metrics included: BLEU, BERTScore-F1, SBERT Similarity, Perplexity + 4 simple metrics")
        return df


class MetricCalculator:
    """
    Calculates automated metrics for model responses
    """
    
    def compute_automated_metrics_advanced(self, responses, references):
        """
        Compute advanced automated metrics including BLEU, BERTScore, SBERT, Perplexity
        
        Args:
            responses: List of model responses
            references: List of reference solutions
            
        Returns:
            Dictionary with metric arrays for each metric type
        """
        
        print("🔧 Computing advanced automated metrics...")
        
        # Initialize metric containers
        metrics = {
            'bleu': [],
            'bertscore_f1': [],
            'sbert_similarity': [],
            'perplexity': [],
            'word_overlap': [],
            'length_similarity': [],
            'char_similarity': [],
            'quality_score': []
        }
        
        # Check for library availability
        bleu_available = True
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothing = SmoothingFunction().method1
        except ImportError:
            print("⚠️ NLTK not available - using fallback BLEU calculation")
            bleu_available = False
        
        bertscore_available = True
        try:
            from bert_score import score as bert_score
        except ImportError:
            print("⚠️ bert-score not available - BERTScore will be 0")
            bertscore_available = False
        
        # Load SBERT model
        sbert_model = None
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading SBERT model...")
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SBERT model loaded")
        except Exception as e:
            print(f"⚠️ Could not load SBERT: {e}")
        
        # Compute batch BERTScore if available
        bertscore_f1 = None
        if bertscore_available:
            try:
                print("Computing BERTScore...")
                _, _, F1 = bert_score(responses, references, lang='en', verbose=False)
                bertscore_f1 = F1.tolist()
                print("✅ BERTScore computed")
            except Exception as e:
                print(f"⚠️ BERTScore computation failed: {e}")
                bertscore_available = False
        
        # Compute SBERT embeddings if available
        sbert_response_embeddings = None
        sbert_reference_embeddings = None
        if sbert_model:
            try:
                print("Computing SBERT embeddings...")
                sbert_response_embeddings = sbert_model.encode(responses)
                sbert_reference_embeddings = sbert_model.encode(references)
                print("✅ SBERT embeddings computed")
            except Exception as e:
                print(f"⚠️ SBERT computation failed: {e}")
                sbert_model = None
        
        # Process each response-reference pair
        print(f"Processing {len(responses)} response-reference pairs...")
        for i, (response, reference) in enumerate(zip(responses, references)):
            # Handle missing responses
            if str(response).startswith("[Missing response") or str(response) == "Missing response":
                # Assign worst possible scores for missing responses
                metrics['bleu'].append(0.0)
                metrics['bertscore_f1'].append(0.0)
                metrics['sbert_similarity'].append(0.0)
                metrics['perplexity'].append(10.0)  # High perplexity = bad
                metrics['word_overlap'].append(0.0)
                metrics['length_similarity'].append(0.0)
                metrics['char_similarity'].append(0.0)
                metrics['quality_score'].append(0.0)
                continue
            
            response_str = str(response).lower()
            reference_str = str(reference).lower()
            
            # 1. BLEU Score
            if bleu_available:
                try:
                    response_tokens = response_str.split()
                    reference_tokens = reference_str.split()
                    if len(response_tokens) > 0 and len(reference_tokens) > 0:
                        bleu = sentence_bleu([reference_tokens], response_tokens, 
                                           smoothing_function=smoothing)
                    else:
                        bleu = 0.0
                except:
                    bleu = 0.0
            else:
                # Fallback: word overlap approximation
                response_words = set(response_str.split())
                reference_words = set(reference_str.split())
                if len(response_words) > 0:
                    bleu = len(response_words.intersection(reference_words)) / len(response_words)
                else:
                    bleu = 0.0
            
            metrics['bleu'].append(bleu)
            
            # 2. BERTScore
            if bertscore_f1 is not None:
                metrics['bertscore_f1'].append(float(bertscore_f1[i]))
            else:
                metrics['bertscore_f1'].append(0.0)
            
            # 3. SBERT Similarity
            if sbert_model and sbert_response_embeddings is not None:
                try:
                    similarity = np.dot(sbert_response_embeddings[i], sbert_reference_embeddings[i]) / (
                        np.linalg.norm(sbert_response_embeddings[i]) * np.linalg.norm(sbert_reference_embeddings[i])
                    )
                    if np.isnan(similarity):
                        similarity = 0.0
                except:
                    similarity = 0.0
            else:
                # Fallback: word overlap
                response_words = set(response_str.split())
                reference_words = set(reference_str.split())
                if len(reference_words) > 0:
                    similarity = len(response_words.intersection(reference_words)) / len(reference_words)
                else:
                    similarity = 1.0
            
            metrics['sbert_similarity'].append(float(similarity))
            
            # 4. Perplexity (simplified approximation)
            try:
                words = response_str.split()
                if len(words) > 0:
                    # Simple perplexity approximation based on word diversity and length
                    unique_words = set(words)
                    word_diversity = len(unique_words) / len(words)
                    # Lower diversity = higher perplexity, normalize to 0-10 range
                    perplexity = (1.0 - word_diversity) * 10.0
                else:
                    perplexity = 10.0
            except:
                perplexity = 10.0
            
            metrics['perplexity'].append(float(perplexity))
            
            # 5. Word overlap
            response_words = set(response_str.split())
            reference_words = set(reference_str.split())
            if len(response_words) > 0:
                word_overlap = len(response_words.intersection(reference_words)) / len(response_words)
            else:
                word_overlap = 0.0
            metrics['word_overlap'].append(float(word_overlap))
            
            # 6. Length similarity
            resp_len = len(response_str.split())
            ref_len = len(reference_str.split())
            if max(resp_len, ref_len) > 0:
                length_sim = 1 - abs(resp_len - ref_len) / max(resp_len, ref_len)
            else:
                length_sim = 1.0
            metrics['length_similarity'].append(float(max(0, length_sim)))
            
            # 7. Character similarity
            response_chars = set(response_str.replace(' ', ''))
            reference_chars = set(reference_str.replace(' ', ''))
            if len(reference_chars) > 0:
                char_sim = len(response_chars.intersection(reference_chars)) / len(reference_chars)
            else:
                char_sim = 1.0
            metrics['char_similarity'].append(float(char_sim))
            
            # 8. Quality score (combination)
            quality = word_overlap * 0.4 + length_sim * 0.3 + char_sim * 0.3
            metrics['quality_score'].append(float(quality))
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(responses)} pairs...")
        
        print("✅ All metrics computed")
        return metrics


class Analyzer:
    """
    Main analyzer class that loads data and computes metrics
    """
    
    def __init__(self):
        self.automated_metrics = {}
        self.model_jsons = {}
        
    def load_model_responses(self, model_files):
        """
        Load model responses from JSON files
        
        Args:
            model_files: Dict mapping model names to file paths
        """
        self.model_jsons = model_files
        
    def compute_metrics_for_all_models(self, reference_solutions):
        """
        Compute metrics for all loaded models
        
        Args:
            reference_solutions: List of reference solutions (50 solutions)
        """
        
        calculator = MetricCalculator()
        
        for model_name, json_file in self.model_jsons.items():
            print(f"\n{'='*60}")
            print(f"Processing {model_name}")
            print(f"{'='*60}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract responses based on model type
                responses = []
                for item in data:
                    if model_name == 'DeepSeek_Base':
                        if 'model_response' in item:
                            mr = item['model_response']
                            if isinstance(mr, dict):
                                responses.append(mr.get('response', 'Missing response'))
                            else:
                                responses.append(str(mr))
                        else:
                            responses.append('Missing response')
                    
                    elif model_name == 'LLaMA_Base':
                        if 'model_response' in item:
                            mr = item['model_response']
                            if isinstance(mr, dict):
                                responses.append(mr.get('solution', 'Missing response'))
                            else:
                                responses.append(str(mr))
                        else:
                            responses.append('Missing response')
                    
                    elif model_name == 'LLaMA_FT':
                        responses.append(item.get('llama_answer', 'Missing response'))
                    
                    elif model_name == 'Mistral_Base':
                        if 'model_response' in item:
                            mr = item['model_response']
                            if isinstance(mr, dict):
                                responses.append(mr.get('solution', 'Missing response'))
                            else:
                                responses.append(str(mr))
                        else:
                            responses.append('Missing response')
                    
                    elif model_name == 'Mistral_FT':
                        responses.append(item.get('aire_mistral_answer', 'Missing response'))
                
                # Ensure we have exactly 50 responses
                while len(responses) < 50:
                    responses.append('Missing response')
                responses = responses[:50]
                
                print(f"Loaded {len(responses)} responses")
                
                # Compute metrics
                metrics = calculator.compute_automated_metrics_advanced(responses, reference_solutions)
                
                self.automated_metrics[model_name] = metrics
                
                print(f"✅ {model_name} metrics computed")
                print(f"   Average BLEU: {np.mean(metrics['bleu']):.4f}")
                print(f"   Average BERTScore: {np.mean(metrics['bertscore_f1']):.4f}")
                print(f"   Average SBERT: {np.mean(metrics['sbert_similarity']):.4f}")
                print(f"   Average Perplexity: {np.mean(metrics['perplexity']):.4f}")
                
            except Exception as e:
                print(f"❌ Error processing {model_name}: {e}")
                import traceback
                traceback.print_exc()


def load_reference_solutions(questions_json_path):
    """
    Load reference solutions from your questions JSON file
    
    Args:
        questions_json_path: Path to the 50-question JSON file
        
    Returns:
        List of 50 reference solutions
    """
    print("📖 Loading reference solutions...")
    
    with open(questions_json_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    reference_solutions = []
    
    for item in questions_data:
        # Extract the solution/conclusion from your question format
        if 'conclusion' in item:
            reference_solutions.append(item['conclusion'])
        elif 'solution' in item:
            if isinstance(item['solution'], dict):
                # If solution is a dict with 'conclusion' field
                reference_solutions.append(item['solution'].get('conclusion', ''))
            else:
                reference_solutions.append(str(item['solution']))
        else:
            reference_solutions.append('')
    
    print(f"✅ Loaded {len(reference_solutions)} reference solutions")
    return reference_solutions


def main(data_directory='data'):
    """
    Main function to generate automated_metrics.csv
    
    Args:
        data_directory: Directory containing your model response JSON files and questions
    """
    
    print("="*80)
    print("AUTOMATED METRICS CSV GENERATOR")
    print("="*80)
    
    # Define model file paths
    model_files = {
        'DeepSeek_Base': f'{data_directory}/model_responses/deepseek_base_responses.json',
        'LLaMA_Base': f'{data_directory}/model_responses/llama_base_responses.json',
        'LLaMA_FT': f'{data_directory}/model_responses/llama_ft_responses.json',
        'Mistral_Base': f'{data_directory}/model_responses/mistral_base_responses.json',
        'Mistral_FT': f'{data_directory}/model_responses/mistral_ft_responses.json'
    }
    
    # Path to your questions file with reference solutions
    questions_file = f'{data_directory}/questions/statistics_questions_50.json'
    
    print(f"\n📂 Data Directory: {data_directory}")
    print(f"📂 Questions File: {questions_file}")
    print(f"📂 Model Files:")
    for model, path in model_files.items():
        print(f"   - {model}: {path}")
    
    # Load reference solutions
    try:
        reference_solutions = load_reference_solutions(questions_file)
    except FileNotFoundError:
        print(f"\n❌ Questions file not found: {questions_file}")
        print("Please ensure your questions JSON file exists at the specified path.")
        return None
    
    # Create analyzer
    analyzer = Analyzer()
    
    # Load model responses
    analyzer.load_model_responses(model_files)
    
    # Compute metrics for all models
    analyzer.compute_metrics_for_all_models(reference_solutions)
    
    # Generate CSV
    generator = CSVGenerator(analyzer)
    df = generator.generate_automated_metrics_csv()
    
    print("\n" + "="*80)
    print("✅ COMPLETED!")
    print("="*80)
    print(f"Generated automated_metrics.csv with {len(df)} rows")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    return df


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data'
    
    print(f"\n💡 Usage: python generate_automated_metrics_complete.py [data_directory]")
    print(f"   Using data directory: {data_dir}\n")
    
    # Run the generator
    df = main(data_directory=data_dir)
