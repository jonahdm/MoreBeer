

import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import re
import requests
import xmltodict as xtd

def get_user_agents_list():
    '''
    Call this function to get a list of valid users agents from https://www.useragentlist.net/
    
    Returns
    -------
    user_agentsL list
        A list of user agents

    '''
    user_agents = []    
    url = 'https://www.useragentlist.net/'
    request = requests.get(url)
    
    user_agents = [t for t in request.text.split('\n') if 'Mozilla/5.0' in t] # Historic artifact makes for easy identification of 
    user_agents = [re.search(r'<strong>(.*?)</strong>', t).group(1) for t in user_agents]
    
    return user_agents


def gen_user_agent_weights(user_agents_list):
    '''
    Generates a list of weights for a given list of user agents, with more favorable agents recieving higher weights.
    Criteria taken from https://scrapfly.io/blog/user-agent-header-in-web-scraping/

    Parameters
    ----------
    user_agents_list : list
        A list of user agents

    Returns
    -------
    user_agents_weights: list
        A list of user agent weights

    '''
    user_agent_weights = [0 for agent in user_agents_list]
    
    for i in range(len(user_agents_list)):
        this_user_agent = user_agents_list[i]
        this_weight = 1000
        
        # Add higher weight based on the browser
        if 'Chrome' in this_user_agent:
            this_weight += 100
        if 'Firefox' or 'Edge' in this_user_agent:
            this_weight += 50
        if 'Chrome Mobile' or 'Firefox Mobile' in this_user_agent:
            this_weight += 0
            
        # Add higher weight based on the OS type
        if 'Windows' in this_user_agent:
            this_weight += 150
        if 'Mac OS X' in this_user_agent:
            this_weight += 100
        if 'Linux' or 'Ubuntu' in this_user_agent:
            this_weight -= 50
        if 'Android' in this_user_agent:
            this_weight -= 100
            
        user_agent_weights[i] += this_weight
    
    return user_agent_weights

def split_and_print_recipes(recipes, output_path_base):
    '''
    Converts a dictionary of recipes generated from BeerXML files into several long-format dataframes, then outputs those dataframes to csv files.
    Current output file types are: metadata, fermetnables, mash, hops, yeast, water, and misc.

    Parameters
    ----------
    recipes : Dict
        DESCRIPTION.
    output_path_base : Str
        The base directory path where files should be printed

    Returns
    -------
    bool
        True if all dataframes were successfully created and printed to csv.

    '''
    # Reminder: Beer-XML units are: kg (mass), L (volumne), C (temperature), min (time), kPa (pressure), and ppm (water chemistry)
    
    full_metadata_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'brewer', 'recipe_type', 'review_score', 'review_count', 'category', 'style_category_number', 'style_letter', 'style_guide', 'style_type', 'batch_size', 'boil_size', 'boil_time',
                                          'estimated_color', 'ibu', 'ibu_method', 'abv', 'og', 'fg', 'co2'])
    full_fermentables_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name',  'fermentable_name', 'fermentables_origin', 'fermentable_type', 'fermentable_amount', 'fermentable_after_boil'])
    full_mash_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'mash_name', 'mash_step', 'mash_type', 'mash_time', 'mash_temp'])
    full_hops_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'hop_name', 'hop_alpha', 'hop_amount', 'hops_use', 'hop_time', 'hopm_form'])
    full_yeast_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'yeast_name', 'yeast_id', 'yeast_lab', 'yeast_type', 'yeast_form', 'yeast_amount', 'yeast_amount_is_weight'])
    full_water_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'water_name', 'calcium', 'bicarbonate', 'sulfate', 'chloride', 'sodium', 'magnesium'])
    full_misc_df = pd.DataFrame(columns = ['recipe_id', 'recipe_name', 'misc_name', 'misc_type', 'misc_use', 'misc_time', 'misc_amount', 'misc_amount_is_weight'])
    
    for this_recipe_id in recipes.keys():
        this_recipe = recipes[this_recipe_id]['RECIPES']['RECIPE']
        this_recipe_name = this_recipe['NAME']
        
        # Recipe Metadata
        this_metadata_df = pd.Series(this_recipe)
        this_metadata_df['recipe_id'] = this_recipe_id
        this_metadata_df.rename(index = {'NAME': 'recipe_name', 'TYPE':'recipe_type', 'BREWER': 'brewer',  'RATING':'review_score',
                                 'NUMBER_OF_REVIEWS':'review_count','BATCH_SIZE':'batch_size', 'BOIL_SIZE':'boil_size', 'BOIL_TIME':'boil_time',
                                 'EST_COLOR':'estimated_color', 'IBU':'ibu', 'IBU_METHOD':'ibu_method', 'EST_ABV':'abv', 'OG':'og',
                                 'FG':'fg', 'CARBONATION_USED':'co2'}, inplace = True) 
        
        this_recipe_style = this_recipe['STYLE']
        this_style_df = pd.Series(this_recipe_style)
        this_style_df.rename(index = {'CATEGORY':'category', 'CATEGORY_NUMBER':'style_category_number', 'STYLE_GUIDE':'style_guide',
                              'STYLE_LETTER':'style_letter', 'TYPE':'style_type',}, inplace = True)
        
      #  if this_metadata_df['ibu_method'] != 'Tinseth': # Different IBU formulas can give wildly different results, so for now we only rely on Tinseth
      #      this_metadata_df['ibu'] = np.NAN   
      #     this_metadata_df['ibu_method'] = np.NAN
    
        this_metadata_df = pd.concat([this_metadata_df, this_style_df], join = 'outer').to_frame().T # Combine the two series and transpose into a DataFrame
        this_metadata_df = this_metadata_df.loc[:, ~this_metadata_df.columns.duplicated()] # Remove duplicate XML Category names that were present in both the Recipe and Style blocks
        this_metadata_df = this_metadata_df[this_metadata_df.columns.intersection(full_metadata_df.columns)] # Pares the current df to only the columns we're interested in
        full_metadata_df = pd.concat([full_metadata_df, this_metadata_df])
        
        # Fermentables Data
        if(this_recipe['FERMENTABLES']):
            if (isinstance(this_recipe['FERMENTABLES']['FERMENTABLE'], dict)): # If this is True, there is only one type of Fermentable in the recipe
                this_fermentables_df = pd.DataFrame([this_recipe['FERMENTABLES']['FERMENTABLE']])
            else:
                this_fermentables_df = pd.DataFrame(this_recipe['FERMENTABLES']['FERMENTABLE'])
            this_fermentables_df.rename(columns= {'NAME':'fermentable_name', 'TYPE':'fermentable_type', 'ORIGIN':'fermentables_origin', 
                                                  'AMOUNT':'fermentable_amount', 'ADD_AFTER_BOIL':'fermentable_after_boil'}, inplace= True)
            this_fermentables_df['recipe_id'] = this_recipe_id
            this_fermentables_df['recipe_name'] = this_recipe_name
            this_fermentables_df = this_fermentables_df[this_fermentables_df.columns.intersection(full_fermentables_df.columns)] # Pares the current df to only the columns we're interested in
            full_fermentables_df = pd.concat([full_fermentables_df, this_fermentables_df])
        
        # Mash Data
        if(this_recipe['MASH']):
            if (isinstance(this_recipe['MASH']['MASH_STEPS'], dict)): # If this is True, there is only one Mash step in this recipe
                this_mash_df = pd.DataFrame([this_recipe['MASH']['MASH_STEPS']])
            else:
                this_mash_df = pd.DataFrame(this_recipe['MASH']['MASH_STEPS'])
            this_mash_df.rename(columns = {'NAME':'mash_name', 'TYPE':'mash_type', 'STEP_TIME':'mash_time', 'STEP_TEMP':'mash_temp'}, inplace = True)
            this_mash_df.insert(0, 'mash_step', range(0, len(this_mash_df))) # Beer XML format doesn't number mash steps, so this is workaround to get it
            this_mash_df['recipe_id'] = this_recipe_id
            this_mash_df['recipe_name'] = this_recipe_name
            this_mash_df = this_mash_df[this_mash_df.columns.intersection(full_mash_df.columns)] # Pares the current df to only the columns we're interested in
            full_mash_df = pd.concat([full_mash_df, this_mash_df])
        
        # Hops Data
        if(this_recipe['HOPS']):
            if (isinstance(this_recipe['HOPS']['HOP'], dict)): # If this is True, there is only one type of Hop in the recipe
                this_hops_df = pd.DataFrame([this_recipe['HOPS']['HOP']])
            else:
                this_hops_df = pd.DataFrame(this_recipe['HOPS']['HOP'])
            this_hops_df.rename(columns = {'NAME':'hop_name', 'ALPHA':'hop_alpha', 'AMOUNT':'hop_amount', 'USE':'hops_use',
                                 'TIME':'hop_time', 'FORM':'hopm_form'}, inplace = True)
            this_hops_df['recipe_id'] = this_recipe_id
            this_hops_df['recipe_name'] = this_recipe_name
            this_hops_df = this_hops_df[this_hops_df.columns.intersection(full_hops_df.columns)] # Pares the current df to only the columns we're interested in
            full_hops_df = pd.concat([full_hops_df, this_hops_df])
            
        # Yeast Data
        if(this_recipe['YEASTS']):
            if (isinstance(this_recipe['YEASTS']['YEAST'], dict)): # If this is True, there is only one type of Yeast in the recipe
                this_yeast_df = pd.DataFrame([this_recipe['YEASTS']['YEAST']])
            else:
                this_yeast_df = pd.DataFrame(this_recipe['YEASTS']['YEAST'])
            this_yeast_df.rename(columns = {'NAME':'yeast_name', 'PRODUCT_ID':'yeast_id', 'LABORATORY':'yeast_lab', 'TYPE':'yeast_type', 'FORM':'yeast_form',
                                  'AMOUNT':'yeast_amount', 'AMOUNT_IS_WEIGHT':'yeast_amount_is_weight'}, inplace = True)
            this_yeast_df['recipe_id'] = this_recipe_id
            this_yeast_df['recipe_name'] = this_recipe_name
            this_yeast_df = this_yeast_df[this_yeast_df.columns.intersection(full_yeast_df.columns)] # Pares the current df to only the columns we're interested in
            full_yeast_df = pd.concat([full_yeast_df, this_yeast_df])
        
        # Misc Data
        if(this_recipe['MISCS']): # MISCS can be empty, depending on the recipe
            if (isinstance(this_recipe['MISCS']['MISC'], dict)): # If this is True, there is only one type of Misc in the recipe
                this_misc_df = pd.DataFrame([this_recipe['MISCS']['MISC']])
            else:
                this_misc_df = pd.DataFrame(this_recipe['MISCS']['MISC'])
            this_misc_df.rename(columns = {'NAME':'misc_name', 'TYPE':'misc_type', 'USE':'misc_use', 'TIME':'misc_time',
                                 'AMOUNT':'misc_amount', 'AMOUNT_IS_WEIGHT':'misc_amount_is_weight'}, inplace = True)
            this_misc_df['recipe_id'] = this_recipe_id
            this_misc_df['recipe_name'] = this_recipe_name
            this_misc_df = this_misc_df[this_misc_df.columns.intersection(full_misc_df.columns)] # Pares the current df to only the columns we're interested in
            full_misc_df = pd.concat([full_misc_df, this_misc_df])
            
        # Water Data
        if(this_recipe['WATERS']):
            if (isinstance(this_recipe['WATERS']['WATER'], dict)): # If this is True, there is only one type of Water profile in the recipe
                this_water_df = pd.DataFrame([this_recipe['WATERS']['WATER']])
            else:
                this_water_df = pd.DataFrame(this_recipe['WATERS']['WATER'])
            this_water_df.rename(columns = {'NAME':'water_name', 'CALCIUM':'calcium', 'BICARBONATE':'bicarbonate', 'SULFATE':'sulfate', 'CHLORIDE':'chloride',
                                  'SODIUM':'sodium', 'MAGNESIUM':'magnesium'}, inplace = True)
            this_water_df['recipe_id'] = this_recipe_id
            this_water_df['recipe_name'] = this_recipe_name        
            this_water_df = this_water_df[this_water_df.columns.intersection(full_water_df.columns)] # Pares the current df to only the columns we're interested in
            full_water_df = pd.concat([full_water_df, this_water_df])
    
        
    # If output files already exist, appends new data to them. Otherwise, writes new files
    full_metadata_df.to_csv(f'{output_path_base}/metadata.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/metadata.csv'))
    full_fermentables_df.to_csv(f'{output_path_base}/fermentables.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/fermentables.csv'))
    full_mash_df.to_csv(f'{output_path_base}/mash.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/mash.csv'))
    full_hops_df.to_csv(f'{output_path_base}/hops.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/hops.csv'))
    full_yeast_df.to_csv(f'{output_path_base}/yeast.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/yeast.csv'))
    full_water_df.to_csv(f'{output_path_base}/water.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/water.csv'))
    full_misc_df.to_csv(f'{output_path_base}/misc.csv', mode = 'a', index = False, header = not os.path.exists(f'{output_path_base}/misc.csv'))
    
    return True

def get_data_from_brewersfriend(output_path_base, chunk_percent = 0.005):
    '''
    The main call function of this script. Iterates across every public page on Brewersfriend pulls each Beer XML file avaialble. Then, outputs to long-format csvs.

    Parameters
    ----------
    output_path_base : str
        The base directory path where files should be printed.
    chunk_percent : float, optional
        What percent of recipes should be used for displaying status and printing checkpoint output. The default is 0.5.

    Returns
    -------
    bool
        True if the entire process succeeded.

    ''' 
    
    user_agents = get_user_agents_list()
    user_agent_weights = gen_user_agent_weights(user_agents)
    
    # Get the number of available recieps from BrewersFriend
    recipe_main_url = 'https://www.brewersfriend.com/homebrew-recipes/'
    first_user_agent = {'User-Agent': random.choices(user_agents, weights = user_agent_weights, k = 1)[0]}
    recipe_main_request = requests.get(recipe_main_url, headers = first_user_agent)
    
    total_recipe_pages = int(re.search(r'class=" recipe_search_nav" data-page-number="(.*?)">Last »</a>', recipe_main_request.text).group(1))
    
    # This is really ugly
    total_recipe_count = re.search(r'[0-9]+Recipesof[0-9]+', re.sub(r'\W+', '', recipe_main_request.text)).group(0) # The total recipe count is displayed as "20 Recipes of ###,###" 
    total_recipe_count = int(re.sub('[0-9]+Recipesof', '', total_recipe_count))
    chunk_size = int(chunk_percent * total_recipe_count)
    
    # Begin data grab of all recipes
    base_xml_url = 'https://www.brewersfriend.com/homebrew/recipe/beerxml1.0'
    base_review_url = 'https://www.brewersfriend.com/homebrew/recipe/view'
    
    current_attempts_count = 0
    current_recipe_count = 0
    skipped_recipe_count = 0
    
    recipes = {}
    
    print(f'Beginning pull of {total_recipe_count} XML recipes.')
    for i in range(13000, total_recipe_pages + 1):
        this_user_agent = {'User-Agent':random.choices(user_agents, weights = user_agent_weights, k = 1)[0]}
        this_recipes_page_url = f'https://www.brewersfriend.com/homebrew-recipes/page/{i}'
    
        this_recipes_request = requests.get(this_recipes_page_url, headers = this_user_agent)
        these_recipe_ids = [t for t in this_recipes_request.text.split('\n') if 'href="/homebrew/recipe/view/' in t]
        these_recipe_ids = [re.search(r'href="/homebrew/recipe/view/(.*?)/', rid) for rid in these_recipe_ids]
        these_recipe_ids = [int(rid.group(1)) for rid in these_recipe_ids if rid is not None]
        
        for rid in these_recipe_ids:
            this_recipe_url = f'{base_xml_url}/{rid}'
            this_review_url = f'{base_review_url}/{rid}'
            recipe_get_success = False
            # Attempt to get the XML format recipe
            this_recipe_request = requests.get(this_recipe_url, headers = this_user_agent)
            this_recipe_content = this_recipe_request.content
        
            try:
                this_recipe_dict = xtd.parse(this_recipe_content)
                recipes[rid] = this_recipe_dict
                current_recipe_count += 1
                current_attempts_count += 1
                recipe_get_success = True
                
            except xtd.expat.ExpatError as xmlE:
                current_attempts_count += 1
                print(f'{rid}: xmlE')
            except Exception as E:
                print(f'Non-XML Format Error Occured while getting recipe {rid}.\n'
                      f'Error is: {E}')
                current_attempts_count += 1
                print(E)
            # Attempt to get the XML format recipe
            if recipe_get_success is True:
                try:
                    this_review_request = requests.get(this_review_url, headers = this_user_agent)
                    this_review_content = str(this_review_request.content)
                    this_review_score = float(re.search(r'<span itemprop="ratingValue">(.*?)</span>', str(this_review_content)).group(1))
                    this_review_count = int(re.search(r'<span itemprop="reviewCount">(.*?)</span>', str(this_review_content)).group(1))
                    
                    this_recipe_dict['RECIPES']['RECIPE']['RATING'] = this_review_score
                    this_recipe_dict['RECIPES']['RECIPE']['NUMBER_OF_REVIEWS'] = this_review_count
                    
                except Exception as E:
                #    print(f'Unable to obtain review score for recipe id {i}.\n'
                #          f'Error is: {E}.\n')
                    this_recipe_dict['RECIPES']['RECIPE']['RATING'] = np.NAN
                    this_recipe_dict['RECIPES']['RECIPE']['NUMBER_OF_REVIEWS'] = np.NAN
                        
            
            if ((current_attempts_count % chunk_size) == 0) or ((i == (total_recipe_pages)) and (rid == these_recipe_ids[-1])): # Every Chunk %, give a progress report, print current output, and clear up some memory
                print(f'Completed {round(100 * current_attempts_count / total_recipe_count, 2)}% of Recipe Requests.')
                split_and_print_recipes(recipes, output_path_base)
                recipes = {}
                
    
    print(f'Attempted to get {current_attempts_count} recipes. Succesfully obtained {current_recipe_count}. Skipped {skipped_recipe_count} recipes.')
    
    # Pare each dataset down to final output
    for this_data_type in ['metadata', 'fermentables', 'mash', 'hops', 'yeast', 'water', 'misc']:
        df = pd.read_csv(f'{output_path_base}/{this_data_type}.csv')
        
        if this_data_type == 'metadata':
            df.drop_duplicates(subset = ['recipe_id']).to_csv(f'{output_path_base}/{this_data_type}.csv', index = False)
        else:
            df.drop_duplicates().to_csv(f'{output_path_base}/{this_data_type}.csv', index = False)

    return True

if __name__ == '__main__':
    print('Beginning data pull from Brewersfriend.')
    
    output_path_base = 'Data/brewersfriend'
    Path(output_path_base).mkdir(parents=True, exist_ok=True)

    if get_data_from_brewersfriend(output_path_base):
        print(f'Data pull successful. Files were printed in {output_path_base}')
    else:
        print('Error occured during processing. Exiting.')
    
    exit()
    