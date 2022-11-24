from matplotlib import pyplot as plt
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
    
    def avg_score_graph(self, base_path, file_name) -> str:
        chunk_size = floor(len(self.episodes) / self.NUMBER_OF_CHUNKS)
        episode_plots = self._list_to_avg_chunks(self.episodes, chunk_size)
        score_plots = self._list_to_avg_chunks(self.scores, chunk_size)
        
        plt.plot(episode_plots, score_plots)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        file_name = f"{base_path}/score_graph_{file_name}.png"
        plt.legend()
        plt.savefig(file_name)
        return file_name
