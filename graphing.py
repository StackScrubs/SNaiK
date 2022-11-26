from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from math import floor
from enum import Enum
from scipy.interpolate import make_interp_spline
import numpy as np

class GraphType(str, Enum):
    AVG = "avg"
    BEST = "best"

class Grapher:
    def __init__(self) -> None:
        self.NUMBER_OF_CHUNKS = 10
            
        self.episodes = []
        self.scores = []
        
    def update(self, episode, score):
        self.episodes.append(episode), 
        self.scores.append(score),
    
    def _reduce_to_avg(self, list: list, chunk_size: int):
        """Divides a long list of values into chunks and finds the average value of each chunk."""
        chunks = [list[i:(i + chunk_size)] for i in range(0, len(list), chunk_size)]
        return [sum(chunks[i]) / len(chunks[i]) for i in range(len(chunks) - 1)]
    
    def _reduce_to_best(self, list: list, chunk_size: int):
        """Divides a long list of values into chunks and finds the best value in each chunk."""
        chunks = [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]
        return [max(chunks[i]) for i in range(len(chunks) - 1)]
    
    def _bullet_list(self, prefix:str, t: dict):
        res = ""
        for i in t.keys():
            if isinstance(t[i], dict):
                res += f"{prefix}{i}:\n" + self._bullet_list("  " + prefix, t[i])
            else:
                res += f"{prefix}{i}: {t[i]}\n"
        return res

    def get_score_graph(self, type, base_path, file_name, card_info) -> str:
        chunk_size = floor(len(self.episodes) / self.NUMBER_OF_CHUNKS)
        
        if type == GraphType.BEST:
            episode_plots = self._reduce_to_best(self.episodes, chunk_size)
            score_plots = self._reduce_to_best(self.scores, chunk_size)
        elif type == GraphType.AVG:
            episode_plots = self._reduce_to_avg(self.episodes, chunk_size)
            score_plots = self._reduce_to_avg(self.scores, chunk_size)
        
        _, ax = plt.subplots()
        at = AnchoredText(
            "Parameters:\n" + self._bullet_list("- ", card_info), 
            prop=dict(size=10), 
            frameon=True, 
            loc='upper left'
        )
        ax.add_artist(at)
        
        X_Y_spline = make_interp_spline(episode_plots, score_plots)
        X_ = np.linspace(np.min(episode_plots), np.max(episode_plots), self.NUMBER_OF_CHUNKS*64)
        Y_ = X_Y_spline(X_)
        
        # plt.plot(episode_plots, score_plots)
        plt.plot(X_, Y_)
        plt.title(f"{type} score over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        file_name = f"{base_path}/{type}_score_graph_{file_name}.png"
        plt.savefig(file_name)
        
        return file_name
    