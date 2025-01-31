import json
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from transformers import AutoTokenizer
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chat.log')
    ]
)

class ChatInterface:
    def __init__(self, model, tokenizer=None, device='cuda', max_length=512):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.conversation_history = []
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
            
        logging.info("ChatInterface initialized successfully")
        logging.info(f"Using device: {device}")
        logging.info(f"Max sequence length: {max_length}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
    def process_query(self, user_input: str, image=None) -> str:
        try:
            logging.info(f"Processing user input: {user_input}")
            
            # Tokenize input
            inputs = self.tokenizer(
                user_input,
                return_tensors='pt',
                max_length=self.max_length // 2,
                truncation=True,
                padding=False
            ).to(self.device)
            
            # Create empty image tensor if no image provided
            if image is None:
                image = torch.zeros((1, 3, 224, 224), device=self.device)
            elif not isinstance(image, torch.Tensor):
                raise ValueError("Image must be a torch tensor")
            
            # Generate response
            try:
                with torch.no_grad():
                    outputs = self.model(image, inputs['input_ids'])
                    # Ensure we extract transformer outputs properly
                    if isinstance(outputs, dict):
                        transformer_outputs = outputs.get('transformer_outputs', {})
                    else:
                        # Handle legacy tuple format temporarily
                        transformer_outputs = outputs[0] if isinstance(outputs, tuple) else {}
                    
                    logits = transformer_outputs.get('text_pred', None)
                    
                    if logits is None:
                        raise ValueError("Text predictions missing")
                    
            except Exception as e:
                logging.error(f"Model forward pass error: {str(e)}")
                logging.error(f"Full traceback: {traceback.format_exc()}")
                return f"Error during model inference: {str(e)}"
            
            # Generate response token by token
            try:
                response_ids = []
                temperature = 0.7
                
                for _ in range(50):
                    last_token_logits = logits[:, -1, :]
                    probs = F.softmax(last_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    if next_token.item() in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        break
                        
                    response_ids.append(next_token.item())
                    
                    # Update input for next iteration
                    new_input = torch.cat([inputs['input_ids'], next_token], dim=1)
                    
                    # Get next prediction
                    outputs = self.model(image, new_input)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        transformer_outputs = outputs.get('transformer_outputs', {})
                    else:
                        transformer_outputs = outputs[0] if isinstance(outputs, tuple) else {}
                    
                    logits = transformer_outputs.get('text_pred', None)
                    
                    if logits is None:
                        raise ValueError("Text predictions missing")
                    
                    if new_input.size(1) >= self.max_length:
                        break
                        
            except Exception as e:
                logging.error(f"Response generation error: {str(e)}")
                logging.error(f"Full traceback: {traceback.format_exc()}")
                return f"Error during response generation: {str(e)}"
            
            # Decode response
            try:
                if response_ids:
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    if len(response_text.strip()) < 2:
                        temperature = 1.0
                        probs = F.softmax(last_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        response_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                else:
                    response_text = "I'm not sure how to respond to that."
                    
                logging.info(f"Generated response: {response_text}")
                
            except Exception as e:
                logging.error(f"Response decoding error: {str(e)}")
                return f"Error during response decoding: {str(e)}"
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_input,
                'assistant': response_text
            })
            
            return response_text
            
        except Exception as e:
            logging.error(f"Unexpected error in process_query: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return f"Unexpected error: {str(e)}"
    
    def save_conversation_history(self, path: str):
        try:
            with open(path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logging.info(f"Saved conversation history to {path}")
        except Exception as e:
            logging.error(f"Error saving conversation history: {e}")
    
    def load_conversation_history(self, path: str):
        try:
            with open(path, 'r') as f:
                self.conversation_history = json.load(f)
            logging.info(f"Loaded {len(self.conversation_history)} previous conversations")
        except Exception as e:
            logging.warning(f"Could not load conversation history: {e}")
    
    def get_conversation_context(self, window_size: int = 5) -> str:
        """Get recent conversation context"""
        recent_history = self.conversation_history[-window_size:]
        context = ""
        for turn in recent_history:
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return context
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logging.info("Cleared conversation history") 