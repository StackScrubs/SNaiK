from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from math import floor
from utils.flatten_dict import flatten_dict
from time import time

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
        chunks = [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]
        return [sum(chunks[i]) / len(chunks[i]) for i in range(len(chunks))]
    
    def _bullet_list(self, prefix: str, info: dict):
        res = ""
        for key in info.keys():
            if isinstance(info[key], dict):
                res += f"{prefix}{key}:\n" + self._bullet_list("  " + prefix, info[key])
            else:
                res += f"{prefix}{key}: {info[key]}\n"
        return res
    
    def avg_score_graph(self, base_path, file_name, card_info: dict) -> str:
        episodes, scores = self.avg_data()

        _, ax = plt.subplots()
        at = AnchoredText(
            "Parameters:\n" + self._bullet_list("- ", card_info), 
            prop=dict(size=10), 
            frameon=True, 
            loc='upper left'
        )
        ax.add_artist(at)
        
        plt.plot(episodes, scores)
        plt.title("Score over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        file_name = f"{base_path}/score_graph_{file_name}.png"
        plt.legend()
        plt.savefig(file_name)
        
        return file_name

    def avg_data(self):
        chunk_size = floor(len(self.episodes) / self.NUMBER_OF_CHUNKS)
        episode_plots = self._reduce_to_avg(self.episodes, chunk_size)
        score_plots = self._reduce_to_avg(self.scores, chunk_size)
        return episode_plots, score_plots        

    def save_stats(self, stats_for: dict):
        from json import dump

        episodes, scores = self.avg_data()
        data = {
            "label": ', '.join([str(x) for x in flatten_dict(stats_for).values()]),
            "episodes": episodes,
            "scores": scores
        }

        file_name = f"{data['label'].replace(', ', '_')}_{time()}.json"

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
