from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from math import floor
from time import time
from enum import Enum
from numba import jit

class StatsType(str, Enum):
    AVG = "avg"
    BEST = "best"

    def __str__(self):
        return self.value

class Grapher:
    def __init__(self) -> None:
        self.NUMBER_OF_CHUNKS = 3
            
        self.episodes = []
        self.scores = []
        
    def update(self, episode, score):
        self.episodes.append(episode), 
        self.scores.append(score),

    def _get_chunk_size(self):
        return floor(len(self.episodes) / self.NUMBER_OF_CHUNKS)

    @staticmethod
    def __chunkize(l: list, chunk_size: int):
        return (l[i:i+chunk_size] for i in range(0, len(l), chunk_size))
    
    @staticmethod
    @jit
    def _simple_moving_average(l: list):
        sample_size = len(l) // 100
        avgs = []
        for i in range(len(l) - (sample_size - 1)):
            sample = l[i:i+sample_size]
            avgs.append(sum(sample) / len(sample))
        return avgs
    
    @staticmethod
    @jit
    def _simple_moving_maximum(l: list):
        sample_size = len(l) // 20
        maxs = []
        for i in range(len(l) - (sample_size - 1)):
            sample = l[i:i+sample_size]
            maxs.append(max(sample))
        return maxs
    
    def __reduce(self, graph_type: StatsType, chunk_size):
        if graph_type == StatsType.BEST:
            maxs = self._simple_moving_maximum(self.scores)
            return self.episodes[:len(maxs)], maxs
        elif graph_type == StatsType.AVG:
            avgs = self._simple_moving_average(self.scores)
            return self.episodes[:len(avgs)], avgs
    
    def _bullet_list(self, prefix:str, info: dict):
        res = ""
        for key in info.keys():
            if isinstance(info[key], dict):
                res += f"{prefix}{key}:\n" + self._bullet_list("  " + prefix, info[key])
            else:
                res += f"{prefix}{key}: {info[key]}\n"
        return res

    def get_score_graph(self, graph_type: StatsType, base_path, card_info) -> str:        
        episodes, scores = self.__reduce(graph_type, self._get_chunk_size())
        
        _, ax = plt.subplots()
        at = AnchoredText(
            "Parameters:\n" + self._bullet_list("- ", card_info), 
            prop=dict(size=10), 
            frameon=True, 
            loc='upper left'
        )
        ax.add_artist(at)
        
        plt.plot(episodes, scores)
        plt.title(f"{graph_type} score over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        file_name = f"{base_path}/{graph_type}_score_graph_{time()}.png"
        plt.savefig(file_name)
        
        return file_name

    @staticmethod
    def _extract_dict_values(in_dict):
        li = []
        if type(in_dict) is not dict:
            return li

        for v in in_dict.values():
            if type(v) == dict:
                li += Grapher._extract_dict_values(v)
            else:
                li.append(v)

        return li

    def save_stats(self, graph_type: StatsType, stats_for: dict):
        from json import dump

        episodes, scores = self.__reduce(graph_type, self._get_chunk_size())
        data = {
            "label": ', '.join([str(x) for x in Grapher._extract_dict_values(stats_for)]),
            "episodes": episodes,
            "scores": scores
        }

        file_name = f"{data['label'].replace(', ', '_')}_{graph_type}_{time()}.json"

        with open(file_name, "w") as f:
            dump(data, f)

        return file_name

def __multi_graph():
    import sys
    from json import load, JSONDecodeError

    args = sys.argv[1:]
    if len(args) < 1:
        print("Invalid number of arguments.", file=sys.stderr)
        print("At least one statistics file must be supplied.", file=sys.stderr)
        exit(-1)
    
    for file in args:
        try:
            with open(file, "r") as f:
                data = load(f)
                plt.plot(data["episodes"], data["scores"], label=data["label"])
                
        except OSError:
            print(f"Failed to read file {file}.", file=sys.stderr)
            exit(-1)
        except JSONDecodeError:
            print("Invalid file format.", file=sys.stderr)
            print("Only supply statistic files saved from main program.", file=sys.stderr)
            exit(-1)
            
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"combined_graph_{time()}.png")

if __name__ == "__main__":
    __multi_graph()
