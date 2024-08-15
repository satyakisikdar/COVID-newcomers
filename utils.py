import pyarrow.parquet as pypq
from pathlib import Path
import pandas as pd
import textwrap 
import time
from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pickle
import ipywidgets as ipw


pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark+presentation'
tqdm.pandas(unit_scale=True)


#unique concepts
works_concepts_unique = pd.read_csv('./data/concepts.csv.gz')
#dictionaries
dict_concept_id_name = works_concepts_unique[['concept_id','concept_name']].set_index('concept_id').to_dict()['concept_name']
dict_concept_name_id = works_concepts_unique[['concept_id','concept_name']].set_index('concept_name').to_dict()['concept_id']
dict_concept_id_level = works_concepts_unique[['concept_id','level']].set_index('concept_id').to_dict()['level']


def read_parquet(path, engine='fastparquet', convert_dtypes=True, **args):
    """
    Read a parquet file
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.name.endswith('.parquet'):
        ## check if a file exists without the extension 
        dir_exists = path.exists()
        if not dir_exists:  # try adding the parquet extension
            if path.with_suffix('.parquet').exists():
                path = path.with_suffix('.parquet')
                
    name = path.stem
        
    print(f'\nReading {name!r} from {str(path)!r} using {engine=}')

    tic = time.time()
    df = pd.read_parquet(path, engine=engine, **args)
    toc = time.time()
    print(f'Read {len(df):,} rows from {path.stem!r} in {toc-tic:.2f} sec.')
    
    if convert_dtypes:
        tic = time.time()
        size_before = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024

        string_cols_d = {}
        for col, dtype in df.dtypes.to_dict().items():
            if dtype == 'object':  # convert object columns to string
                string_cols_d[col] = 'string[python]'
            if col == 'type' or col == 'concept_name':
                if dtype != 'category':
                    string_cols_d[col] = 'category'
            if col == 'publication_month':
                if dtype != 'uint8':
                    string_cols_d[col] = 'uint8'
            if col == 'score':
                if dtype != 'float16':
                    string_cols_d[col] = 'float16'
                
        df = df.astype(string_cols_d) 
        
        size_after = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024
        toc = time.time()
        print(f'Converting dtypes took {toc-tic:.2f} sec. Size before: {size_before:.2f}GB, after: {size_after:.2f}GB')
    display(df.head(3))
    return df


def peek_parquet(path):
    """
    peeks at a parquet file (or a directory containing parquet files) and prints the following:
    * Path
    * schema
    * number of pieces (fragments)
    * number of rows 
    """
    if isinstance(path, str):
        path = Path(path)
        
    parq_file = pypq.ParquetDataset(path)
    piece_count = len(parq_file.fragments)
    schema = textwrap.indent(parq_file.schema.to_string(), ' '*4)
    row_count = sum(frag.count_rows() for frag in parq_file.fragments)
    
    st = [
        f'Name: {path.stem!r}',  
        f'Path: {str(path)!r}',
        f'Files: {piece_count:,}',
        f'Rows: {row_count:,}',
        f'Schema:\n{schema}',
        f'5 random rows:',
    ]
    print('\n'.join(st))
    sample_df = parq_file.fragments[0].head(5).to_pandas()  # read 5 rows from the first fragment
    display(sample_df)

    return


def dump_pickle(obj, path):
    if isinstance(path, str):
        path = Path(path)
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)
    print(f'Pickle saved at {str(path)!r}')
    return
    

def load_pickle(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.exists()
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def get_topic_info(topic:str, ow_duration: int = 3, tw_duration: int = 5, verbose=True):
    """
    get topic_ids, prior experience topic ids, 
    tw_duration: training window duration in years
    ow_duration: observation window duration in years
    """
    topic_ = topic.replace('-noJan1', '')
    if topic_ == 'COVID':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_524204448':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_116675565':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_89623803':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')
    
    elif topic_ == 'COVID_107130276':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {89623803, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_PH':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448, 138816342}  # public health 
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_MED_PH':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {71924100, 138816342}  # medicine and public health only
        start_date_ow = pd.to_datetime('2020-01-01')
          
    elif topic_ == 'DL':  # deep learning
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}    # transformers: 66322947
        topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2017-01-01') 
        
    elif topic_ == 'DL_2020':  # deep learning but starts in 2020
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}  
        topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2020-01-01')
        
    elif topic_ == 'DL_ML':  # deep learning with ML 
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}  
        topic_ids_exp = {119857082}
        # topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2017-01-01') 
    
    elif topic_ == 'StatPhy':
        topic_ids = {121864883}
        topic_ids_exp = {149288129, 51329190, 202213908, 29912722, 99874945, 151342819}
        start_date_ow = pd.to_datetime('2017-01-01') 

    elif topic_ == 'StatMech':
        topic_ids = {99874945}
        topic_ids_exp = {121864883}
        start_date_ow = pd.to_datetime('2017-01-01') 
    
    elif topic_ == 'QED':  # quantum electrodynamics
        topic_ids = {3079626}       # quantum electrodynamics
        topic_ids_exp = {62520636}  # quantum mechanics
        start_date_ow = pd.to_datetime('2017-01-01') 
        
    elif topic_ == 'Cryptography':
        topic_ids = {203062551}     # cryptography
        topic_ids_exp = {38652104}  # computer security
        start_date_ow = pd.to_datetime('2017-01-01') 
     
    elif topic_ == 'H1N1':  
        topic_ids = {2910793863, 3017774372, 2777546802}  # C2777546802 is influenza type A 
        # topic_ids = {2910793863, 3017774372}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2009-01-01')
        
    elif topic_ == 'HIV':  
        topic_ids = {2780195530, 3013748606}  
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('1981-01-01')
        
    elif topic_ == 'Ebola':  
        topic_ids = {2777469322,2909623084,2910344980,2778933410}   # 22 is Ebola virus, 84 is Ebolavirus, 980: ebola hemorrhagic, 10: ebola vaccine
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2014-01-01')
        
    elif topic_ == 'Zika':  
        topic_ids = {2777053367}  
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2015-01-01')
        
    elif topic_ == 'MERS': 
        topic_ids = {2776525042,2777691041, 2777648638, 2778137277}   # 38: Coronavirus, 77: Betacoronavirus
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2012-01-01')
        
    elif topic_ == 'SARS':  
        topic_ids = {3020799909, 2778859668, 3007834351, 2777648638, 2778137277}  # C2778859668: SARS 
        # topic_ids = {3020799909, 2778859668}  # C2778859668: SARS 
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2002-01-01')
        
    else:
        raise NotImplementedError(f'{topic}')

    # end_date_ow = start_date_ow + pd.DateOffset(years=ow_duration)
    end_date_ow = pd.to_datetime(f'{start_date_ow.year + 2}-12-31')
    observation_window = pd.date_range(start_date_ow, end_date_ow, freq='MS')

    end_date_tw = start_date_ow  # end date of exposure window is the start date of OW
    start_date_tw = start_date_ow - pd.DateOffset(years=tw_duration)
    end_date_tw = end_date_tw - pd.DateOffset(days=1) # end one day earlier 
    training_window = pd.date_range(start_date_tw, end_date_tw, freq='MS')

    if verbose:
        print(f'{topic!r} Core topics')
        for id in topic_ids:
            print(f'\t{id}, {dict_concept_id_name[id]} (level {dict_concept_id_level[id]})')
    
        print(f'\n{topic!r} Prior experience topics:')
        for id in topic_ids_exp:
            print(f'\t{id}, {dict_concept_id_name[id]} (level {dict_concept_id_level[id]})')
        
        print(f'Training window: {start_date_tw.to_period("D")} to {end_date_tw.to_period("D")}')
        print(f'Observation window: {start_date_ow.to_period("D")} to {end_date_ow.to_period("D")}')
    
    return topic_ids, topic_ids_exp, start_date_tw, end_date_tw, start_date_ow, end_date_ow


def all_paths_exist(*paths):
    """
    returns True only if all the paths exist
    """
    return all(p.exists() for p in paths)


def filter_dataframes_timerange(whole_works, whole_works_authors, whole_works_concepts, start_date, end_date, exclude_solo_authors=True, verbose=False):
    """
    Filter the dataframes based on a time range
    add some print statements
    """
    if verbose:
        print(f'Filtering dataframes from {start_date.date()} to {end_date.date()}')
    works_filt = (
        whole_works
        .query('publication_date.between(@start_date, @end_date)')
    )
    
    if exclude_solo_authors:
        works_filt = (
            works_filt
            .query('num_authors>1')
        )
    # print(f'{len(works_ow):,} works, discarding solo works')
    work_ids = set(works_filt.index)
    
    works_authors_filt = (
        whole_works_authors
        .query('work_id.isin(@work_ids)')
    )
    # print(f'{works_authors_ow.author_id.nunique():,} authors')

    works_concepts_filt = (
        whole_works_concepts
        .query('work_id.isin(@work_ids)')
    )
    # print(f'{len(works_concepts_ow):,} works concept tags')

    return works_filt, works_authors_filt, works_concepts_filt


def combine_and_assign_author_classes(df):
    """
    assign phase and author classes to dfs
    """
    if isinstance(df, list):
        df = pd.concat(df)
        
    df = (
        df
        .assign(
            date=lambda df_: pd.to_datetime(df_.date),
        )
    )
    
    if 'work_id' in df.columns and 'num_experts' in df.columns:
        df = (
            df
            .assign(
                frac_expert=lambda df_: df_.num_experts.div(df_.num_authors),
                class_=lambda df_ : pd.cut(
                    df_.frac_expert,
                    bins=[0, 0.000001, 0.5, 0.999, 1],
                    include_lowest=True,
                    labels=['0%', '1-50%', '51-99%', '100%'],
                ),
            )
        )
        
    return df
    

def get_monthly_works_authors_stats(
    topic, month_idx, date, 
    topic_works_authors, topic_newcomers_month, 
    topic_bellwethers_month,
    topic_experts_month, topic_repeat_authors_month,
):
    """
    get work stats for topics work authors 
    """
    monthly_works_author_stats = (
        topic_works_authors
        .groupby('work_id', sort=False, as_index=False)
        .agg({
            'work_id': 'first',
            'author_id': [
                'count', 
                lambda x: len(set(x) & topic_newcomers_month), 
                lambda x: len(set(x) & topic_bellwethers_month), 
                lambda x: len(set(x) & topic_experts_month),
                lambda x: len(set(x) & topic_repeat_authors_month),
            ]
        })
        .droplevel(0, axis=1)
        .rename(columns={'first': 'work_id', 'count': 'num_authors', 
                         '<lambda_0>': 'num_newcomers', '<lambda_1>': 'num_bellwethers', 
                         '<lambda_2>': 'num_experts', '<lambda_3>': 'num_repeats'})
        .assign(month_idx=month_idx, topic=topic, date=date)
    )
    return monthly_works_author_stats


def plot_yearly_works_authors(works_authors, all_covid_works, all_outbreak_science_works):
    combined_yearly_counts = pd.concat([
        works_authors
        .query('work_id.isin(@all_covid_works)')
        .groupby('publication_year', as_index=False)
        .agg(
            works=('work_id', 'nunique'),
            authors=('author_id', 'nunique')
        )
        .assign(topic='COVID-19'),
        works_authors
        .query('work_id.isin(@all_outbreak_science_works)')
        .groupby('publication_year', as_index=False)
        .agg(
            works=('work_id', 'nunique'),
            authors=('author_id', 'nunique')
        )
        .assign(topic='Outbreak Science')
    ])
    fig = (
        combined_yearly_counts
        .plot
        .barh
        (
            y='publication_year', x=['authors', 'works'],
            facet_col='topic', 
            barmode='group', log_x=True, height=600,
            text_auto='.2s', 
            labels={'publication_year': '', 'value': 'Log Counts'},
        )
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    fig.add_hline(y=2019.5, line_dash='dot', annotation_text='Training Window', 
              annotation_position='bottom right')
    # fig.update_layout()
    return fig


def plot_author_types_timeline(author_types_df):
    """
    Plot the author types in the OW
    """
    agg_stats_long = (
        pd.melt(author_types_df, id_vars=['topic', 'date'], value_vars=['newcomers_frac', 'bellwethers_frac', 'experts_frac'], var_name='author_type')
    )
    fig = (
        px.area(
            agg_stats_long, x='date', y='value', color='author_type',
            category_orders={'author_type': ['experts_frac', 'bellwethers_frac', 'newcomers_frac']},
            text='value', labels={'date': '', 'value': 'Authors %'},
            title=f'Author percentage split for COVID-19 articles',
            height=600,
        )
    )
    fig.add_vline(x='2020-03-30', line_dash='dash')
    fig.update_traces(textposition='bottom center', texttemplate = '%{text:.0%}', textfont_size=12)
    
    fig.update_yaxes(tickformat=',.0%')
    fig.update_xaxes(
        dtick='M1',
        ticklabelmode='period',
        range=['2020-1-15', '2023-01-15'],
        tickformat='%b\n%Y',
    )
    
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='right',
        x=1
    ))
    return fig

    
def plot_work_types_timeline(work_types_df):
    """
    Plot the work types in the OW
    """
    topic = work_types_df.topic.values[0]
    
    x, y = 'date', 'works_frac'
    groupby_col = 'date'
    plot_kind = 'area'
    plot_args = dict()

    text_auto = '.0%'
    s = 'percentage'
    title = f'Work percentage split for COVID-19 articles'
    height = 600
    
    fig = (
        work_types_df
        .plot
        (
            x=x, y=y, color='class_', text=y,
            category_orders=dict(class_=['0%', '1-50%', '51-99%', '100%']),
            height=height, title=title,
            color_discrete_sequence=['lightblue', 'plum', 'gold', 'crimson'],
            labels={'date': '', 'works_frac': 'Works %', 'class_': ''},
            kind=plot_kind, **plot_args,
        )
    )
    fig.add_vline(x='2020-03-30', line_dash='dash')
    fig.update_traces(textposition='bottom center', texttemplate='%{text:.0%}', textfont_size=12)
    
    fig.update_layout(uniformtext_mode='hide',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.7,
    ))
    if 'COVID' in topic or '2020' in topic:
        xrange = ['2019-12-15', '2023-01-15']
    else:
        xrange = ['2016-12-15', '2020-01-15']

    fig.update_xaxes(
        dtick='M1',
        ticklabelmode='period',
        range=['2020-1-15', '2023-01-15'],
        tickformat='%b\n%Y',
    )
    fig.update_yaxes(tickformat=',.0%')
    return fig 
