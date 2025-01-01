import re
from tqdm import tqdm
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

def must_find_task_type(str1,str2):
    for t in ['weight','logp','complexity','topological polar surface area','homo-lumo gap','homo','lumo','scf energy','docking']:
        if (t in str1.lower()) or (t in str2.lower()):
            return t,True
    raise ValueError('task type not found. Please check manully.')
def visualize_performance(gen, gt, task_name='logp'):
    """
    Visualizes the performance metrics for the given generated and ground truth values.
    
    Parameters:
    gen (list): List of generated values.
    gt (list): List of ground truth values.
    task_name (str): Name of the task, used for plot titles. Default is 'logp'.
    """
    # Filter out None values
    gen = [item for item in gen if item is not None]
    gt = [item for item in gt if item is not None]
    
    if len(gen) != len(gt):
        raise ValueError("The length of generated values and ground truth values must be the same.")
    
    # Convert lists to numpy arrays for calculations
    gen = np.array(gen)
    gt = np.array(gt)
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(gt, gen, alpha=0.5)
    plt.title(f'Scatter plot of Generated vs Ground Truth {task_name}')
    plt.xlabel('Ground Truth')
    plt.ylabel('Generated')
    plt.grid(True)
    plt.show()

def validate_numbers(gen_num, gt_num,task_type,exclude_outliers=True):
    if len(gen_num)>=1 and len(gt_num)>=1:
        gen_num, gt_num = gen_num[0], gt_num[0]
    else:
        return None, None
    try:
        gen_num, gt_num = float(gen_num), float(gt_num)
    except:
        return None, None
    if exclude_outliers:
        if task_type in ['homo-lumo gap','homo','lumo']:
            if not (-20<gen_num<20):# and -20<gt_num<20):
                return None,None
        elif task_type in ['scf energy']:
            if not (-5<gen_num<0 ):#and -5<gt_num<0):
                return None,None
        elif task_type in ['logp']:
            if not (-30<gen_num<50):# and -30<gt_num<50):
                return None,None
        elif task_type in ['topological polar surface area']:
            if not (0<=gen_num<2000):# and 0<=gt_num<2000):
                return None,None
        elif task_type in ['complexity']:
            if not (0<=gen_num<10000):# and 0<=gt_num<10000):
                return None,None
        elif task_type in ['weight']:
            if not (0<gen_num<4000):# and 0<gt_num<4000):
                return None,None 
        elif task_type in ['docking']:
            pass
        else:
            raise ValueError(f"Invalid task type {task_type}")
    return gen_num,gt_num
    

def eval_number(input_file,exclude_outliers=1,just_see=None):
    with open(input_file, 'r') as f:
        lines=f.readlines()
    if just_see:
        lines=lines[:just_see]
    if 'pubchemqc' in input_file or 'pubchemqa' in input_file:
        gen={
        'homo':[],'lumo':[],'homo-lumo gap':[],'scf energy':[]
        }
        gt={k:[] for k in gen.keys()}
    elif 'docking' in input_file:
        gen={
            'docking':[]
        }
        gt={k:[] for k in gen.keys()}
    else:
        gen={
            'weight':[],'logp':[],'topological polar surface area':[],'complexity':[],
        }
        gt={k:[] for k in gen.keys()}
    
    #get numbers in file, and see how many are valid
    pattern=r'-?\d+\.\d+' 
    for line in tqdm(lines):
        item=line.strip().split("<iamsplit>")
        assert len(item)==2
        task_type,flag=must_find_task_type(item[0],item[1])
        if not flag:
            gen[task_type].append(None)
            gt[task_type].append(None)
            continue
        gen_num = re.findall(pattern, item[0])
        gt_num = re.findall(pattern,item[1])
        
        gen_num,gt_num=validate_numbers(gen_num,gt_num,task_type,exclude_outliers=exclude_outliers)

        gen[task_type].append(gen_num)
        gt[task_type].append(gt_num)
    for k in gen.keys():
        assert len(gen[k])==len(gt[k])
        print(f"{k}: valid ratio is {1-gen[k].count(None)/len(gen[k])}; total is {len(gen[k])}")
    #print('----------------------------------------')
    
    #statistics and visualization
    return_maes=[]
    return_r2s=[]
    for k in gen.keys():
        if k.lower()=='scf energy':
            gen_temp=[10*item for item in gen[k] if item is not None]
            gt_temp=[10*item for item in gt[k] if item is not None]
        else:
            gen_temp=[item for item in gen[k] if item is not None]
            gt_temp=[item for item in gt[k] if item is not None]
        #Calculate MAE
        assert len(gen_temp)==len(gt_temp)
        mae=sum(abs(y_t - y_p) for y_t, y_p in zip(gt_temp, gen_temp)) / len(gen_temp)
        #print(f"{k}: MAE value is {mae:.4f}")
        
        # Calculate R²
        gt_temp = np.array(gt_temp)
        gen_temp = np.array(gen_temp)
        ss_res = np.sum((gt_temp - gen_temp) ** 2)
        ss_tot = np.sum((gt_temp - np.mean(gt_temp)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        #print(f"{k}: R² value is {r2:.4f}")

        # Calculate Pearson correlation coefficient
        mean_gen_temp = np.mean(gen_temp)
        mean_gt_temp = np.mean(gt_temp)
        pearson = np.sum((gen_temp - mean_gen_temp) * (gt_temp - mean_gt_temp)) / np.sqrt(np.sum((gen_temp - mean_gen_temp) ** 2) * np.sum((gt_temp - mean_gt_temp) ** 2))
        #print(f"{k}: Pearson value is {pearson:.4f}")
        
        #Equal
        equal=[1 if abs(y_t - y_p)<1e-8 else 0 for y_t, y_p in zip(gt_temp, gen_temp)]
        equal_ratio=mean(equal)
        #print(f"{k}: Equal ratio is {equal_ratio:.4f}")
        
        return_maes.append(mae)
        return_r2s.append(r2)
        
        #计算RMSE
        rmse = np.sqrt(np.mean((gt_temp - gen_temp) ** 2))
        print(f"{k}: RMSE value is {rmse:.4f}")
    return(return_maes+return_r2s)


if __name__ == "__main__":
    results=eval_number('path/to/HME/generated_result.txt',exclude_outliers=1,just_see=None)
    results=[str(round(item,4)) for item in results]
    print('&'.join(results[:4]),'\n','&'.join(results[4:]))
