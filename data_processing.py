import pandas as pd
import pathlib
from os import listdir
from os.path import isfile, join
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List

@dataclass
class ProcessedData:
    cap: bool
    classes: list
    data: pd.DataFrame
    cpu_usage_mean: float
    ram_usage_mean: float
    gpu_usage_mean: float
    gpu_memory_mean: float
    cycle_time_mean: float

log_foldername= "logs"
output_data_foldername= "logs_processed"
output_data_foldername_cap = output_data_foldername + "/cap"
output_data_foldername_uncap = output_data_foldername + "/no_cap"

def get_datas_from_files(files, file_folder):
    
    datas = []

    for f in files:

        data = pd.read_csv(file_folder / f, skiprows=2)

        

        # get classes
        description = pd.read_csv(file_folder / f, nrows=2, header=None)
        classes_list = description.at[0, 0].split("-")[1:]


        # get cap
        cap = description.at[1, 0].split(":")[1]

        data_dict = {}

        # store metadata
        data_dict = {
            "metadatas": {
                "classes": classes_list,
                "cap": cap
            },
            "data": data
        }

        datas.append(data_dict)
    
    return datas
    

def process(datas: List[dict]):

    pdata_cap = []
    pdata_uncap = []

    for data_dict in datas:

        data = data_dict["data"]

        # format data
        data[" CPU_Usage (%)"] = data[" CPU_Usage (%)"].replace("%", "", regex=True).astype(float)
        data[" RAM_Usage (%)"] = data[" RAM_Usage (%)"].replace("%", "", regex=True).astype(float)
        data[" cycle_time (s)"] = data[" cycle_time (s)"].replace("s", "", regex=True).astype(float)
        data[" GPU_usage (%)"] = data[" GPU_usage (%)"].replace(["N\\A", "N/A"], "0", regex=True).fillna("0").astype(float)
        data[" GPU_memory_usage"] = data[" GPU_memory_usage"].replace(["N\\A", "N/A"], "0", regex=True).fillna("0").astype(float)

        # calculate means
        cpu_usage_mean = data[" CPU_Usage (%)"].mean()
        ram_usage_mean = data[" RAM_Usage (%)"].mean()
        gpu_usage_mean = data[" GPU_usage (%)"].astype(float).mean()
        gpu_memory_mean = data[" GPU_memory_usage"].astype(float).mean()
        cycle_time_mean = data[" cycle_time (s)"].mean()

        # store processed data
        pdata = ProcessedData(
            cap=data_dict["metadatas"]["cap"],
            classes=data_dict["metadatas"]["classes"],
            data=data,
            cpu_usage_mean=cpu_usage_mean,
            ram_usage_mean=ram_usage_mean,
            gpu_usage_mean=gpu_usage_mean,
            gpu_memory_mean=gpu_memory_mean,
            cycle_time_mean=cycle_time_mean
        )

        if pdata.cap == " True":
            pdata_cap.append(pdata)
        else:
            pdata_uncap.append(pdata)

    return pdata_cap, pdata_uncap

def create_dataframe_of_means(data_ext):
    # create dataframe of means
    dataframes = []
    for data in data_ext:

        dict_buffer = {'N. classes': len(data.classes),
                        'CPU Usage (%)': data.cpu_usage_mean,
                        'RAM Usage (%)': data.ram_usage_mean,
                        'GPU Usage (%)': data.gpu_usage_mean,
                        'GPU Memory Usage': data.gpu_memory_mean,
                        'Cycle Time (s)': data.cycle_time_mean}
        dataframe_buffer = pd.DataFrame(dict_buffer, index=[0])
        dataframes.append(dataframe_buffer)

    # concatenate dataframes based on the number of classes
    result_df = pd.concat(dataframes, ignore_index=True).sort_values(by='N. classes').reset_index(drop=True)
    print(result_df)

    return result_df

def save_result_csv(result_dataframe, output_data_foldername, sufix=""):
    # Save the result dataframe as CSV in a folder
    output_folder = pathlib.Path(__file__).parent.resolve() / output_data_foldername
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / ("performance_log_processed " + sufix + " .csv")
    result_dataframe.to_csv(output_file, index=False)

def save_plots(current_folder, output_data_foldername, result_df, sufix=""):
    
    result_df.plot(x='N. classes', y='CPU Usage (%)', kind='line', marker='o')
    title = 'CPU Usage Over Number of Classes' + sufix
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('CPU Usage (%)')
    output_folder = current_folder / output_data_foldername /  "images" 
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / (title + ".png"))
    

    result_df.plot(x='N. classes', y='GPU Usage (%)', kind='line', marker='o')
    title = 'GPU Usage Over Number of Classes' + sufix
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('GPU Usage (%)')
    plt.savefig(output_folder / (title + ".png"))

    result_df.plot(x='N. classes', y='RAM Usage (%)', kind='line', marker='o')
    title = 'RAM Usage Over Number of Classes' + sufix
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('RAM Usage (%)')
    plt.savefig(output_folder / (title + ".png"))

    result_df.plot(x='N. classes', y='GPU Memory Usage', kind='line', marker='o')
    title = 'GPU Memory Usage Over Number of Classes' + sufix
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('GPU Memory Usage')
    plt.savefig(output_folder / (title + ".png"))

    result_df.plot(x='N. classes', y='Cycle Time (s)', kind='line', marker='o')
    title = 'Cycle Time Over Number of Classes' + sufix
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('Cycle Time (s)')
    plt.savefig(output_folder / (title + ".png"))



def save_plots2(current_folder, output_data_foldername, result_df_cap, result_df_uncap):
    
    plt.figure()

    # result_df_cap.plot(x='N. classes', y='CPU Usage (%)', kind='line', marker='o', label='Cap')
    # result_df_uncap.plot(x='N. classes', y='CPU Usage (%)', kind='line', marker='o', label='Uncap')

    # CPU Usage
    plt.plot(result_df_cap['N. classes'], result_df_cap['CPU Usage (%)'], 
             marker='o', label='Cap', color='blue')
    plt.plot(result_df_uncap['N. classes'], result_df_uncap['CPU Usage (%)'], 
             marker='o', label='Uncap', color='red')
    title = 'CPU Usage Over Number of Classes'
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('CPU Usage (%)')
    plt.legend()

    output_folder = current_folder / output_data_foldername /  "images" 
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / (title + ".png"))
    plt.clf()

    # GPU Usage
    plt.plot(result_df_cap['N. classes'], result_df_cap['GPU Usage (%)'], 
             marker='o', label='Cap', color='blue')
    plt.plot(result_df_uncap['N. classes'], result_df_uncap['GPU Usage (%)'], 
             marker='o', label='Uncap', color='red')
    title = 'GPU Usage Over Number of Classes'
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('GPU Usage (%)')
    plt.legend()
    plt.savefig(output_folder / (title + ".png"))
    plt.clf()

    # RAM Usage
    plt.plot(result_df_cap['N. classes'], result_df_cap['RAM Usage (%)'], 
             marker='o', label='Cap', color='blue')
    plt.plot(result_df_uncap['N. classes'], result_df_uncap['RAM Usage (%)'], 
             marker='o', label='Uncap', color='red')
    title = 'RAM Usage Over Number of Classes'
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('RAM Usage (%)')
    plt.legend()
    plt.savefig(output_folder / (title + ".png"))
    plt.clf()

    # GPU Memory Usage
    plt.plot(result_df_cap['N. classes'], result_df_cap['GPU Memory Usage'], 
             marker='o', label='Cap', color='blue')
    plt.plot(result_df_uncap['N. classes'], result_df_uncap['GPU Memory Usage'], 
             marker='o', label='Uncap', color='red')
    title = 'GPU Memory Usage Over Number of Classes'
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('GPU Memory Usage')
    plt.legend()
    plt.savefig(output_folder / (title + ".png"))
    plt.clf()

    # Cycle Time
    plt.plot(result_df_cap['N. classes'], result_df_cap['Cycle Time (s)'], 
             marker='o', label='Cap', color='blue')
    plt.plot(result_df_uncap['N. classes'], result_df_uncap['Cycle Time (s)'], 
             marker='o', label='Uncap', color='red')
    title = 'Cycle Time Over Number of Classes'
    plt.title(title)
    plt.xlabel('Number of Classes')
    plt.ylabel('Cycle Time (s)')
    plt.legend()
    plt.savefig(output_folder / (title + ".png"))
    plt.clf()
    

def process_data():

    current_folder = pathlib.Path(__file__).parent.resolve()
    file_folder = pathlib.Path(__file__).parent.resolve() / log_foldername


    # get all files in the folder
    files = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]

    print("loaded files:")
    for f in files: print(f)
    print("")

    # get all data from the files
    datas = []

    datas = get_datas_from_files(files, file_folder)

    (pdata_cap, pdata_uncap) = process(datas)

    
    # process to cap nd uncap
    result_df_cap = create_dataframe_of_means(pdata_cap)
    save_result_csv(result_df_cap, output_data_foldername_cap, "_cap")
    save_plots(current_folder, output_data_foldername_cap, result_df_cap, "_cap")

    result_df_uncap = create_dataframe_of_means(pdata_uncap)
    save_result_csv(result_df_uncap, output_data_foldername_uncap)
    save_plots(current_folder, output_data_foldername_uncap, result_df_uncap)

    save_plots2(current_folder, output_data_foldername, result_df_cap, result_df_uncap)




if __name__ == "__main__":
    process_data()