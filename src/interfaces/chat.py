import json
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from transformers import AutoTokenizer

class ChatInterface:
    def __init__(self, model, tokenizer_name='bert-base-uncased', device='cuda'):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.conversation_history = []
        self.max_length = 512
        
    def process_query(self, user_input: str, image=None) -> str:
        # Tokenize input
        inputs = self.tokenizer(
            user_input,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        ).to(self.device)
        
        # Prepare batch
        batch = {
            'text_ids': inputs['input_ids'],
            'text_mask': inputs['attention_mask']
        }
        
        if image is not None:
            batch['image'] = image.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model(batch, task='generate')
            
            # Get response from policy logits
            policy_logits = outputs['policy_logits']
            response_tokens = torch.argmax(policy_logits, dim=-1)
            
            # Decode response
            response = self.tokenizer.decode(response_tokens[0], skip_special_tokens=True)
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response
        })
        
        return response
    
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