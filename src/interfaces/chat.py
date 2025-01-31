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
            
            # Check if model has been trained
            if not hasattr(self.model, 'transformer') or not hasattr(self.model.transformer, 'text_head'):
                return "I need to be trained before I can respond to messages. Please run the training process first."
            
            # Tokenize input with special tokens
            inputs = self.tokenizer(
                user_input,
                return_tensors='pt',
                max_length=self.max_length // 2,  # Leave room for response
                truncation=True,
                padding=False  # Don't pad yet
            ).to(self.device)
            
            # Create empty image tensor if no image provided
            if image is None:
                image = torch.zeros((1, 3, 224, 224), device=self.device)
                logging.info("Created empty image tensor")
            elif not isinstance(image, torch.Tensor):
                raise ValueError("Image must be a torch tensor")
            
            # Generate response token by token
            response_ids = []
            current_ids = inputs['input_ids']
            current_mask = inputs['attention_mask']
            
            for _ in range(50):  # Maximum 50 new tokens
                # Prepare batch
                batch = {
                    'text_ids': current_ids,
                    'text_mask': current_mask,
                    'image': image.to(self.device) if image is not None else None
                }
                
                # Generate next token
                with torch.no_grad():
                    outputs = self.model(batch, task='generate')
                    if 'transformer_outputs' not in outputs or 'text_pred' not in outputs['transformer_outputs']:
                        break
                        
                    logits = outputs['transformer_outputs']['text_pred']
                    if logits is None or logits.size(0) == 0:
                        break
                    
                    # Get the last token's prediction
                    last_token_logits = logits[:, -1, :]
                    
                    # Apply temperature sampling
                    temperature = 0.7
                    probs = F.softmax(last_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Stop if we predict an end token
                    if next_token.item() in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        break
                        
                    # Add to response
                    response_ids.append(next_token.item())
                    
                    # Update current input
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    current_mask = torch.ones_like(current_ids)
                    
                    # Stop if getting too long
                    if current_ids.size(1) >= self.max_length:
                        break
            
            # Decode the response
            if response_ids:
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                if len(response_text.strip()) < 2:  # If response too short, try again with higher temperature
                    temperature = 1.0
                    probs = F.softmax(last_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    response_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            else:
                response_text = "I'm not sure how to respond to that."
                
            # Update conversation history
            self.conversation_history.append({
                'user': user_input,
                'assistant': response_text
            })
            
            return response_text
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            return "I encountered an error while processing your request."
    
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