'''draw quadrant chart'''
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import MultipleLocator
from torchvision.utils import make_grid, save_image
def drew_lineChart_similarity_two_plot(df  , df2, save_path ,legent_position ='upper right'):

    plt.figure(figsize=(12, 5))
    # 第一個 subplot：左邊
    plt.subplot(1, 2, 1)



    df = df.rename(columns={'reduce_percentage': 'percent reduction'})
    df = df.rename(columns={'value': 'Similarity with M (FID)'})
    
    # 設定文字大小
    plt.xlabel('Percent Reduction', fontsize=16)  # 設定 X 軸標籤文字大小為 14
    plt.ylabel('Similarity with M (FID)', fontsize=16)  # 設定 Y 軸標籤文字大小為 14
    # 調整 X 和 Y 軸上數字的大小
    plt.tick_params(axis='x', labelsize=16)  # 設定 X 軸上數字的大小為 12
    plt.tick_params(axis='y', labelsize=16)  # 設定 Y 軸上數字的大小為 12
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))  # 設置 x 軸主要刻度為整數

    sns.lineplot(x="percent reduction", y="Similarity with M (FID)",
             hue="type",
             data=df)
    # 添加格線
    plt.grid(True)
    # 设置 legend 的位置和大小
    #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16) 
    plt.legend(loc='upper left', fontsize=15) 
    # 第二個 subplot：右邊
    plt.subplot(1, 2, 2)
    
    df2 = df2.rename(columns={'reduce_percentage': 'percent reduction'})
    df2 = df2.rename(columns={'value': 'Similarity with M (FID)'})
    # 設定文字大小
    plt.xlabel('Percent Reduction', fontsize=16)  # 設定 X 軸標籤文字大小為 14
    plt.ylabel('Similarity with M (FID)', fontsize=16)  # 設定 Y 軸標籤文字大小為 14
    # 調整 X 和 Y 軸上數字的大小
    plt.tick_params(axis='x', labelsize=16)  # 設定 X 軸上數字的大小為 12
    plt.tick_params(axis='y', labelsize=16)  # 設定 Y 軸上數字的大小為 12
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))  # 設置 x 軸主要刻度為整數

    sns.lineplot(x="percent reduction", y="Similarity with M (FID)",
             hue="type",
             data=df2)
    # 添加格線
    plt.grid(True)
    # 设置 legend 的位置和大小
    #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16) 
    plt.legend(loc='upper left', fontsize=15) 
  


    # 調整布局
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')  # 保存图形到文件中
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口   

def drew_lineChart_similarity(df , save_path ,legent_position ='upper right'):

   




    df = df.rename(columns={'reduce_percentage': 'percent reduction'})
    df = df.rename(columns={'value': 'Similarity with M (FID)'})
    
    # 設定文字大小
    plt.xlabel('Percent Reduction', fontsize=16)  # 設定 X 軸標籤文字大小為 14
    plt.ylabel('Similarity with M (FID)', fontsize=16)  # 設定 Y 軸標籤文字大小為 14
    # 調整 X 和 Y 軸上數字的大小
    plt.tick_params(axis='x', labelsize=16)  # 設定 X 軸上數字的大小為 12
    plt.tick_params(axis='y', labelsize=16)  # 設定 Y 軸上數字的大小為 12
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))  # 設置 x 軸主要刻度為整數

    sns.lineplot(x="percent reduction", y="Similarity with M (FID)",
             hue="type",
             data=df)
    # 添加格線
    plt.grid(True)
    # 设置 legend 的位置和大小
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16) 
    
    plt.savefig(save_path, bbox_inches='tight')  # 保存图形到文件中
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口   

def draw_quadrant_chart(df , loss_type ='Mse' , chart_type = "gather",length_limit=(150,150),xory_line =None , isX_axis =True ,save_path = None,no_label_icon=False):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    # 繪製散點圖
    x_length_limit = length_limit[0]
    y_length_limit = length_limit[1]
    hue_colors = {
        'LILE': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
        'LIHE': (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
        'HIHE': (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
        'HILE': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
    }
    if no_label_icon == True :
        style_mark = {
                        '0': 'o',
                        '1': 'o',
                        '2': 'o',
                        '3': 'o',
                        '4': 'o',
                        '5': 'o',
                        '6': 'o',
                        '7': 'o',
                        '8': 'o',
                        '9': 'o',
                    }
    else:
        style_mark = {
                        '0': 'o',
                        '1': 'X',
                        '2': 's',
                        '3': 'P',
                        '4': 'D',
                        '5': '>',
                        '6': '^',
                        '7': 'p',
                        '8': 'v',
                        '9': '*',
                    }
    if chart_type =='gather':
        '''draw chart'''
        
        sns.scatterplot(data=df, x=loss_type, y="KLD", hue="quadrant_"+loss_type+"_KLD", palette=hue_colors , markers=style_mark)
        # 根據輸入在 X 軸劃一條紅色的線
        if xory_line  is not None:
            if isX_axis == True :
                # drew a red line
                pass
                #plt.axvline(x=xory_line[0], color='red', linestyle='--')
                #plt.axvline(x=xory_line[1], color='red', linestyle='--')
            else :
                pass
                #plt.axhline(y=xory_line[0], color='red', linestyle='--')
                #plt.axhline(y=xory_line[1], color='red', linestyle='--')
        # 固定 X 軸和 Y 軸的範圍
        plt.xlim(0, x_length_limit)  # 替換 x_min 和 x_max 分別為您想要固定的 X 軸範圍的最小值和最大值
        plt.ylim(0, y_length_limit)  # 替換 y_min 和 y_max 分別為您想要固定的 Y 軸範圍的最小值和最大值
        plt.xlabel('MSE', fontsize=15)  # 設定 X 軸標籤文字大小為 14
        plt.ylabel('KLD', fontsize=15)  # 設定 Y 軸標籤文字大小為 14
        plt.tick_params(axis='x', labelsize=15)  # 設定 X 軸上數字的大小為 12
        plt.tick_params(axis='y', labelsize=15)  # 設定 Y 軸上數字的大小為 12
        # 設置 legend 的位置和大小
        plt.legend(loc='upper right', fontsize=15,title="") 
        #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small') 
        # 顯示圖形
        plt.savefig(save_path+'/gather_plot.png', bbox_inches='tight')  # 保存图形到文件中
        #plt.show()
    elif chart_type == 'separate':
        g = sns.FacetGrid(df, col="label", hue="quadrant_"+loss_type+"_KLD", palette=hue_colors)
        g.map(sns.scatterplot, loss_type, "KLD", alpha=.7)
        # 使用 set 方法來設置 X 軸和 Y 軸的範圍
        g.set(xlim=(0, x_length_limit), ylim=(0, y_length_limit))  
        g.add_legend()
        
        plt.savefig(save_path+'/separate_plot.png')  # 保存图形到文件中
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口
    
def drew_quadrant_countBylabel(df , _flag_name , save_path= None , persentage = False):
    
    if persentage == False:
        df_output   = pd.DataFrame({'label':[],'      LILE      ' : [] , '      LIHE      ':[] , '      HIHE      ':[] , '      HILE      ':[]  })
    else:
        df_output   = pd.DataFrame({'label':[],'      LILE      ' : [] , '      LIHE      ':[] , '      HIHE      ':[] , '      HILE      ':[]  })
    for i in range(10):
        
        if persentage == False:
            df_empty   = pd.DataFrame({'label':[],'      LILE      ' : [] , '      LIHE      ':[] , '      HIHE      ':[] , '      HILE      ':[]  })
            flag_name =_flag_name # 'quadrant_Mse_KLD' # 'quadrant_Dssim_KLD'
            df_label = df[df['label'] == str(i)]
            df_empty['label']      = [i]
            df_empty['      LILE      '] = [df_label[df_label[flag_name] == 'LILE'][flag_name].count()]
            df_empty['      LIHE      '] = [df_label[df_label[flag_name] == 'LIHE'][flag_name].count()]
            df_empty['      HIHE      '] = [df_label[df_label[flag_name] == 'HIHE'][flag_name].count()]
            df_empty['      HILE      '] = [df_label[df_label[flag_name] == 'HILE'][flag_name].count()]
            df_output   = pd.concat([df_output, df_empty])
        else :
            
            df_empty   = pd.DataFrame({'label':[],'      LILE      ' : [] , '      LIHE      ':[] , '      HIHE      ':[] , '      HILE      ':[]  })
            flag_name =_flag_name # 'quadrant_Mse_KLD' # 'quadrant_Dssim_KLD'
            df_label = df[df['label'] == str(i)]
            label_count = int(df_label['label'].count())
            df_empty['label']      = [i]
            qua1_persent =(df_label[df_label[flag_name] == 'LILE'][flag_name].count()/label_count)
            qua2_persent =(df_label[df_label[flag_name] == 'LIHE'][flag_name].count()/label_count)
            qua3_persent =(df_label[df_label[flag_name] == 'HIHE'][flag_name].count()/label_count)
            qua4_persent =(df_label[df_label[flag_name] == 'HILE'][flag_name].count()/label_count)

            qua1_persent = round((qua1_persent)*100 , 2)
            qua2_persent = round((qua2_persent)*100 , 2)
            qua3_persent = round((qua3_persent)*100 , 2)
            qua4_persent = round((qua4_persent)*100 , 2)
            df_empty['      LILE      '] = [ str(qua1_persent)+'%']
            df_empty['      LIHE      '] = [str(qua2_persent)+'%']
            df_empty['      HIHE      '] = [str(qua3_persent)+'%']
            df_empty['      HILE      '] = [str(qua4_persent)+'%']
            df_output   = pd.concat([df_output, df_empty])     

    df_output   =df_output.reset_index(drop =True)

    if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 假设 df 是你的 DataFrame 数据
    df = df_output

    # 创建一个子图
    fig, ax = plt.subplots()

    # 隐藏坐标轴
    ax.axis('off')

    # 绘制表格
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # 设置表格的字体大小
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 调整表格布局
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # 保存表格为图片
    plt.savefig(save_path+'/quadrantChaet_table.png')
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口



'''顯示特定區間的KLD或Loss,可選一次顯示的數量 
percentage代表 -> 數值 * ％
1. select_value_type = 'Mse'  or 'Dssim'
2. ValuesPercentage =(0 , 1 ) that means from 0% to 1% .
'''
def show_quadrant_image_selectByValuesPercentage(df,x_df ,x_hat_df ,quadrant_data_type, _distance_part_number ,image_save_path, select_value_type = 'Mse'\
                                                 ,ValuesPercentage =(0 , 1 ),changePercentToNumber= False  , changeTo_KLD_limit= False):
    process_df = df 
    distance_part_number = _distance_part_number
    
    if changeTo_KLD_limit == True :
        filer_value_type ='KLD'
        sort_type = select_value_type
        name = 'KLD'
    else :
        filer_value_type =select_value_type
        sort_type = 'KLD'
        name = 'MSE'
    Max_length    = df[filer_value_type].max()    
    '''選取區間在 lower ~ upper 之間'''
    if changePercentToNumber == True :
        count_perset = int(df[sort_type].count() *ValuesPercentage[0])
        sorted_dfKLD = df.sort_values(by=sort_type)
        sorted_dfKLD = sorted_dfKLD.reset_index(drop=True)
        lower_limit= sorted_dfKLD.loc[count_perset, sort_type]


        count_perset = int(df[sort_type].count() *ValuesPercentage[1])
        
        upper_limit= sorted_dfKLD.loc[count_perset, sort_type]
        #lower_limit   = ValuesPercentage[0]
        #upper_limit   = ValuesPercentage[1]

    else :
        lower_limit   = Max_length *(ValuesPercentage[0])  /100
        upper_limit   = Max_length *(ValuesPercentage[1]) /100
    print(f'lower_limit :{lower_limit} ')
    print(f'upper_limit :{upper_limit} ')
    name = name+'_'+str(round(lower_limit,1))+'_'+str(round(upper_limit,1))
    '''篩選 df資料'''
    process_df = process_df[process_df[ filer_value_type ] <= upper_limit]
    process_df = process_df[process_df[ filer_value_type ] >= lower_limit]
    print(f' df len:{len(process_df)} ')
    
    '''排序'''
    process_df = process_df.sort_values(by=sort_type, ascending=True)
    
    '''draw charts and images'''
    '''在範圍內均勻間隔取樣'''
    
    
    # 如果数据点数量不足 interval，则加入最后一个数据点
    if len(process_df) != 0  :
        threshold_upper    = df[sort_type].max()
        threshold = 0    
        count = 0
        '''計算間隔'''
        interval = ((threshold_upper)/ distance_part_number)
        print(f'interval :{interval}')
        
        while threshold < threshold_upper:
            
            threshold += interval
            temp_df = process_df[process_df[ sort_type ] <= threshold+interval]
            temp_df = temp_df[temp_df[ sort_type ] >= threshold]
            #print(f'temp_df len :{len(temp_df)}')
            if len(temp_df) == 0:
                continue 
            temp_df = temp_df.iloc[0]
            '''第一次建立'''
            if count == 0 :
                charts_display = pd.DataFrame([temp_df], columns=process_df.columns)
                count += 1
            else :
                charts_display = pd.concat([charts_display, pd.DataFrame([temp_df], columns=process_df.columns)])
        print(f'charts_display type :{type(charts_display)}')
    if select_value_type == 'Mse' and len(charts_display) != 0 :
        draw_quadrant_chart(charts_display , loss_type =select_value_type , chart_type = "gather",length_limit=(150,150),xory_line=(lower_limit,upper_limit) , isX_axis = not changeTo_KLD_limit\
                            ,save_path = os.path.join(image_save_path,name ),no_label_icon=True)
    elif select_value_type == 'Dssim'and len(charts_display) != 0:
        draw_quadrant_chart(charts_display  , loss_type =select_value_type , chart_type = "gather",length_limit=(1,150),xory_line=(upper_limit,upper_limit), isX_axis = not changeTo_KLD_limit\
                            ,save_path = os.path.join(image_save_path,name ))
        
    
    
    '''draw images input x '''
    x_df_quadrant = x_df.iloc[charts_display['x_index'].values]
    print(f'x_df_quadrant len :{len(x_df_quadrant)}')
    x_df_quadrant = torch.tensor(x_df_quadrant.values)
    quaList =charts_display[quadrant_data_type].tolist()
    KLDList =charts_display['KLD'].tolist()
    MseList =charts_display[select_value_type].tolist()
    display_images_in_rows(x_df_quadrant.cpu().view(-1,28,28) ,quaList ,MseList\
                               ,KLDList, select_value_type ,distance_part_number,display_type ='newline',save_path=os.path.join(image_save_path,name\
                                                                                                                                ,'x_df.png' ))
    '''draw images input x_hat '''
    x_df_quadrant = x_hat_df.iloc[charts_display['x_index'].values]
    x_df_quadrant = torch.tensor(x_df_quadrant.values)
    quaList =charts_display[quadrant_data_type].tolist()
    KLDList =charts_display['KLD'].tolist()
    MseList =charts_display[select_value_type].tolist()
    display_images_in_rows(x_df_quadrant.cpu().view(-1,28,28) ,quaList ,MseList\
                               ,KLDList, select_value_type ,distance_part_number,display_type ='newline',save_path=os.path.join(image_save_path,name\
                                                                                                                                ,'x_hat_df.png' )) 


def display_images_in_rows(image_list ,qua_list , mse_list , kld_list ,data_name,   _images_per_row=5 ,turn_off_valuse_display=False ,display_type ='no_newline',save_path=None):
    
    num_images = len(image_list)
    if num_images == 0 :
        return 0
    if display_type =='no_newline':
        images_per_row = _images_per_row  # 每行显示的图像数量
    else :
        images_per_row = 5
    num_rows = (num_images + images_per_row - 1) // images_per_row  # 计算所需的行数

    if num_images == 0:
        raise ValueError("The number of images must be at least 1.")

    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows),
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.3})

    for i in range(num_images):
        row = i // images_per_row
        col = i % images_per_row
        ax = axes[row, col] if num_rows > 1 else axes[col]  # 处理只有一行的情况
        ax.imshow(image_list[i], cmap='gray')
        if turn_off_valuse_display:
            ax.set_title(f'number :{i }  ', fontsize=10)
        else:            
            ax.set_title(f'qua :{qua_list[i] } ,{data_name} {round(mse_list[i] ,2)},KLD :{round(kld_list[i] ,2) }  ', fontsize=10)
        ax.axis('off')
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整wspace和hspace以设置水平和垂直间距
    #plt.show()
    plt.savefig(save_path)  # 保存图形到文件中
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口   

def save_grid_img(dispersion_images ,save_path_filename =os.path.join('Global_explanable_framework','outputs', '{}.png'.format(type))
                  ):
    grid = make_grid(dispersion_images.view(-1, 1, 28, 28), nrow=10, normalize=True)
    images = torch.stack([ grid], dim=0).cpu()
    save_image(tensor=images,
                               fp=save_path_filename,
                               nrow=10, pad_value=1)
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口

def save_grid_dataframe(dispersion_dataframe,save_path_filename =os.path.join('Global_explanable_framework','outputs', 'quadrantChaet_table_{}.png'.format(type))):
    df = dispersion_dataframe
    # 创建一个子图
    fig, ax = plt.subplots(figsize=(10, 6))  # 调整图片大小为宽10英寸，高6英寸

    # 隐藏坐标轴
    ax.axis('off')

    # 绘制表格
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # 设置表格的字体大小
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 调整表格布局
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # 保存表格为图片
    plt.savefig(save_path_filename , bbox_inches='tight')
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口