from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from math import floor

class Grapher:
    def __init__(self) -> None:
        self.NUMBER_OF_CHUNKS = 10
            
        self.episodes = []
        self.scores = []
        
    def update(self, episode, score):
        self.episodes.append(episode), 
        self.scores.append(score), 
    
    def _list_to_avg_chunks(self, list, chunk_size):
        chunks = [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]
        return [sum(chunks[i]) / len(chunks[i]) for i in range(len(chunks))]
    
    def _bullet_list(self, prefix:str, t: dict):
        res = ""
        for i in t.keys():
            if isinstance(t[i], dict):
                res += f"{prefix}{i}:\n" + self._bullet_list("  " + prefix, t[i])
            else:
                res += f"{prefix}{i}: {t[i]}\n"
        return res
    
    def avg_score_graph(self, base_path, file_name, card_info) -> str:
        chunk_size = floor(len(self.episodes) / self.NUMBER_OF_CHUNKS)
        episode_plots = self._list_to_avg_chunks(self.episodes, chunk_size)
        score_plots = self._list_to_avg_chunks(self.scores, chunk_size)
        
        _, ax = plt.subplots()
        at = AnchoredText(
            "Parameters:\n" + self._bullet_list("- ", card_info), 
            prop=dict(size=10), 
            frameon=True, 
            loc='upper left'
        )
        ax.add_artist(at)
        
        plt.plot(episode_plots, score_plots)
        plt.title("Score over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        file_name = f"{base_path}/score_graph_{file_name}.png"
        plt.legend()
        plt.savefig(file_name)
        
        return file_name
