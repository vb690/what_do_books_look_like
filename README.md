<video src="page/videos/league.mp4" width="700" height="700" controls preload> </video>  
 
This project aims to visualize the representation learned by a word-level language model trained on a collection of classics of the literature. The aim is purely aesthetic and not scientific, what presented here should be interpreted with caution (or not intepreted at all).

This project was inspired by [What do numbers look like](https://johnhw.github.io/umap_primes/index.md.html).

## Motivation

## Features

* Automated data preparation: from PDF to numpy arrays.
* Integrated hyper-parameters tuning.
* Possibility to grid-serach UMAP hyper-parameters.

## How to Use

1. Create a folder in `data/raw` named `your_project_name`.
2. Populate the the `your_project_name` with the books you want to embed in PDF format.
3. In `data/jsons` create `your_project_name.json` mapping the title of each book to a valid `matplolib` colormap.
```python
{
  "Dracula": "Reds",
  "The Picture of Dorian Gray": "plasma",
  "Strange Case of Dr Jekyll and Mr Hyde": "viridis",
  "King Solomon's Mines": "autumn",
  "Twenty Thousand Leagues Under the Sea": "winter",
  "The Invisible Man": "summer"
}
```
4. From the terminal, launch `run_pipeline.py` and specify `your_project_name` when propted to do so.  
  
Alternatively, each script in `run_pipeline.py` can be launched separately (in case a specific step needs to be executed in isolation)
  
5. When the script is done (this can take quite some time), use the notebook `generate_visuals.ipynb` for obtaining the visuals.

## The League of Extraordinary Gentlemen

<p float="center">
  <img src="page/images/dracula.png" width="300" />
  <img src="page/images/king_solomon_mines.png" width="300" /> 
  <img src="page/images/the_invisible_man.png" width="300" />
  <img src="page/images/the_picture_of_dorian_gray.png" width="300" />
  <img src="page/images/the_strange_case.png" width="300" /> 
  <img src="page/images/twenty_thousand_leagues_under_the_sea.png" width="300" />
</p>

## Credits

## License 
[The MIT License](https://github.com/vb690/what_do_books_look_like/blob/master/LICENSE)

