# MoreBeer 
MoreBeer is an in-progress project which utilizes machine learning to generate new beer recipes for homebrewing.

## Current Features

### get_recipe_data
Obtains over 280,000 user-submitted homebrew recipes publicaly available from [BrewerFriend](https://www.brewersfriend.com/homebrew-recipes/). These are obtained as [BeerXML](http://www.beerxml.com/) files which are lightly processed, split, and saved into CSV files.
Due to concerns with repository size and the privacy of BrewersFriend users, this input data is saved in this repo. An archive of these recipe files can be shared upon request.

### generate_recipe_names
Utilizes the GPT-2 Transformer architecture demonstrated in Andrej Karpathy's [MakeMore](https://github.com/karpathy/makemore/) to create a model trained on recipe names.
This model is then able to generate completely new recipe names. Some personal favorite example names include:
```
Czech Juice
Amarillo Whut
Bitter to Bitter Fle
Milk Cram Sweeteiss
Boston BY19
Noddbrown Ale
1197 Witbier
```
The best version of each model type created is saved as a PyTorch file under Models/NamesGenerator. Optimization is still ongoing.

## Usage
This code should be relatively plug-and-play, as long as all packages are properly installed. Do note that obtaining the recipe data will take a very long run time, and care should be taken with any deployment so as not to overload BrewersFriends wtih too many requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
* [Andrej Karpathy](https://karpathy.ai/) for his incredible resources and tutorials regarding machine learning.
* [BrewerFriend](https://www.brewersfriend.com/homebrew-recipes/) for hosting the world's greatest collection of homebrew recipes.

## To every data scientist, homebrewer, or anyone somewhere inbetween - cheers! :beer:
