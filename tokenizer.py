from transformers import Wav2Vec2CTCTokenizer
import re

class Wav2Vec2WordpieceTokenizer(Wav2Vec2CTCTokenizer):
    def __init__(self, vocab_file, **kwargs):
        Wav2Vec2CTCTokenizer.__init__(self, vocab_file, **kwargs)
        self.special_cases = set(['gia', 'qui', 'que', 'qua'])
        
    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        output_tokens = []
        for token_idx, token in enumerate(text.split()):
            sub_tokens = []
            last_token = None
            if token[:3] in self.special_cases:
                last_token = token[:2]
                token = token[2:]
                
            end = len(token)
            while end > 0:
                start = 0
                cur_substr = None
                while start < end:
                    substr = token[start:end]
                    if substr in self.encoder:
                        cur_substr = substr
                        break
                    start += 1
                if cur_substr is None:
                    sub_tokens.insert(0, self.unk_token)
                    end = start - 1
                else:
                    sub_tokens.insert(0, cur_substr)
                    end = start
            
            if token_idx > 0:
                output_tokens.append(self.word_delimiter_token)
            
            if last_token:
                output_tokens.append(last_token)
            output_tokens.extend(sub_tokens)
        return output_tokens

    def tokenize(self, text, **kwargs):
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        tokenized_text = self._tokenize(text)
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text