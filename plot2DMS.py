import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import pandas as pd
import os
import glob
import scipy.signal as sci

class For2Dplot():
    a = [8, 15]
    b = [1, 2, 5, 6, 9, 11, 13, 18]
    #DIR = 'C:\\googledrive\\export\\test\\connect_csv\\'
    def __init__(self):
        self.xname = 'time'
        self.yname = 'count'
        self.autorange = True
    def valleyDetect(self,baseArray,threshold=0.15):
        sig_mins = sci.argrelmin(baseArray, order=1)
        place0 = baseArray < threshold
        indices_base = np.nonzero(place0[1:] != place0[:-1])[0] + 1
        indices = np.union1d(indices_base,sig_mins[0])
        indices = np.insert(indices,0,0)
        result = []
        for i in np.arange(0,len(indices),1):
            try:
                if threshold < baseArray[indices[i]] :
                    pass
                    #print(i,"pass1")
                else:
                    range =baseArray[indices[i]:indices[i+1]]
                    #print("range",indices[i],indices[i+1],range)
                    minList = np.argwhere(range == np.amin(range)) + indices[i]
                    minMax = np.max(minList)
                    minMin = np.min(minList)
                    if minMax == minMin:
                        #print(i,range,minMin)
                        result.append(minMax)
                    else:
                        #print(i,range,minMax,minMin)
                        result.append(minMax)
                        result.append(minMin)
            except:
                if threshold < baseArray[indices[i]] :
                    pass
                    #print(i,"pass2")
                else:
                    #print(i,baseArray[indices[i]:],np.argmin(baseArray[indices[i]:])+indices[i])
                    result.append(np.argmin(baseArray[indices[i]:])+indices[i])
        #print(result)
        try:
            result = np.union1d(result,sig_mins[0])
        except:
            result = sig_mins[0].tolist()
            print("none")
        #plt.scatter(result,baseArray[result])
        #plt.scatter(sig_mins,baseArray[sig_mins])
        #print(baseArray[result])
        return result

    def get_peak_left_and_right(self,peak=a,bottom=b):
        length_peak = len(peak)
        length_bottom = len(bottom)
        j = 0
        right = []
        left = []
        peaks = []
        for i in np.arange(0,length_bottom,1):
            if bottom[i] < peak[j] :
                try:
                    if bottom[i+1] >peak[j]:
                        left.append(int(bottom[i]))
                        right.append(int(bottom[i+1]))
                        peaks.append(int(peak[j]))
                        j += 1
                        if j == length_peak:
                            j=0
                            break
                        else:
                            pass
                    else:
                        pass
                except :
                    break
            elif bottom[i] >= peak[j]:
                try:
                    while bottom[i] >= peak[j]:
                        j += 1
                    if j == length_peak:
                        j = 0
                        break
                    else:
                        pass
                except:
                    break
            else:
                if j == length_peak:
                    j = 0
                    break
                else:
                    pass
        peak_and_range = (left,right,peaks)
        return peak_and_range

    def read_csv2visualive(self,DIR,base=None,num=None):
        os.chdir(DIR)
        file_names = glob.glob('./*CSV')
        file_names = glob.glob('./*csv')
        all_data = []
        all_name= []
        print(file_names)
        resultDf = pd.DataFrame()
        for n ,name in enumerate(file_names):
            print(name)
            df = pd.read_csv(name,header=0,engine='python')
            df = df.set_index("time")
            #print(df)
            if (base==None):
                pass
                df = (df / df.sum()) * 100000
            else:
                dfBase = pd.read_csv(base, header=0, engine='python')
                dfBase= dfBase.set_index("time")
                #print(dfBase)
                df = df -dfBase
                df[df < 0] = 0
                df = (df / df.sum()) * 100000
            df = df.reset_index()
            df = df.fillna(0)
            df = df.astype(int)

            df["time"] = df["time"]/20
            #df = df.set_index(["time"])
            name = name.replace(".\\", "")
            name = name.replace("csv", "")
            name = name.replace("CDF", "")
            name = name.replace("TXT", "")
            name = name.replace("..", "")
            name = name.replace(".", "")
            if (num == None):
                total_data = df.sum(axis=1)
            else:
                num = str(num)
                #num = '"' + num  +'"'
                total_data = df[num]
            #Total ion
            total_data = total_data.fillna(0)
            #total_data = total_data[500:8000]

            # 谷検出
            #sig_mins = sci.argrelmin(total_data.values, order=1)
            #sig_mins_list = sig_mins[0].tolist()
            sig_mins_list = self.valleyDetect(total_data.values,400)
            #ピーク検出
            # min_dist = 60
            # sig_peaks = peakutils.indexes(total_data, thres=0.001,min_dist=min_dist)
            # sig_peaks_list = sig_peaks.tolist()

            #ピーク検出別パターン
            sig_peaks = sci.argrelmax(total_data.values, order=4)
            sig_peaks_list = sig_peaks[0].tolist()

            print(len(sig_peaks_list),len(sig_mins_list))
            #検出量の調整
            # while  len(sig_mins_list) < len(sig_peaks_list):
            #     min_dist = min_dist + 20
            #     sig_peaks = peakutils.indexes(total_data, thres=0.005, min_dist=min_dist)
            #     sig_peaks_list = sig_peaks.tolist()
            #     print(len(sig_peaks_list))

            #谷でピークを挟み込み
            all= self.get_peak_left_and_right(sig_peaks_list,sig_mins_list)
            print("valley",len(all[0]),len(all[1]),len(all[2]))
            # peaks_x = np.round(df.index.values[sig_peaks],2).tolist()
            # print(peaks_x)
            # peaks_y = total_data.values[sig_peaks].tolist()
            # peaks_y_upp = (100+total_data.values[sig_peaks]).tolist()
            sig_mins = all[0] + all[1]
            print("result",sig_mins)
            #sig_mins = []
            cave_pos_x = np.round(df.index.values[sig_mins],2).tolist()
            print("next")
            cave_pos_y = total_data.values[sig_mins].tolist()
            sum_df = pd.DataFrame(columns=["time","area","index","left","right"])
            #面積計算
            for i in np.arange(0,len(all[0])-1,1):
                print(i)
                peak_area = int(sum(total_data[all[0][i]:all[1][i]]))
                peak_time = df["time"][all[2][i]]
                left = all[0][i]
                right =all[1][i]
                #peak_time = round(df.index.values[all[2][i]]/600,2)
                index = all[2][i]
                if peak_area < 100:
                    pass
                else:
                    df_temp = pd.DataFrame([[peak_time,peak_area,index,left,right]],columns=["time","area","index","left","right",])
                    sum_df = sum_df.append(df_temp)
            save_sum_df = sum_df[["time","area"]]
            save_sum_df['mz'] = [num] * len(save_sum_df.index)
            save_sum_df = save_sum_df.set_index(["time","mz"])

            sum_df = sum_df.set_index("index")
            #データ保存

            saveName =DIR + "\\area\\" + name +"Area.csv"
            #save_sum_df.to_csv(saveName)
            resultDf = pd.concat([resultDf, save_sum_df], axis = 1, ignore_index = True)
            #ピーク位置
            peaks_x = round(df["time"][sum_df.index.values],2)
            peaks_y_name = sum_df["time"]
            try:
                peaks_y_pos = total_data.values[sum_df.index.values ].tolist()
            except:
                peaks_y_pos =[]
            #積分値
            area_x = peaks_x
            area_y_name = sum_df["area"]
            try:
                area_y_pos = (total_data.values[sum_df.index.values]).tolist()
            except:
                area_y_pos = []
            #谷の値
            try:
                left = df["time"][sum_df["left"]].tolist()
                right= df["time"][sum_df["right"]].tolist()
                cave_pos_x = left+right
            except:
                left,right,cave_pos_x =[],[],[]
            try:
                left_right = sum_df["left"].append(sum_df["right"],ignore_index=True).tolist()
            except:
                left_right =[]
            print(left_right)
            #cave_pos_name = round((cave_pos_x),2)
            try:
                cave_pos_y =  total_data.values[left_right]
            except:
                cave_pos_y = []
            #以下、2次元プロットの描画
            #本体
            data = go.Scatter(
                # time
                x=df["time"],
                # y = df[name],
                # 合計値
                y=total_data,
                name=name,
                line=dict(
                    width=2)
            )
            #Peak描画
            trace = go.Scatter(
                x= peaks_x,
                y = peaks_y_pos,
                mode='markers+text',
                text = peaks_x,
                name = 'Peak',
                textposition='top left',
                textfont=dict(
                    size=15
                ),
                marker=dict(
                    size=3,
                    color='rgb(100,0,255)',
                    symbol=6
                ),
            )
            #面積の描画
            trace_area = go.Scatter(
                x= area_x,
                y = area_y_pos,
                mode='markers+text',
                text = area_y_name,
                name = 'area',
                textposition='top right',
                textfont=dict(
                    size=15
                ),
                marker=dict(
                    size=1,
                    color='rgb(100,0,0)',
                    symbol=2
                ),
            )
            #谷の描画（min）
            trace_min = go.Scatter(
                x=cave_pos_x,
                y=cave_pos_y,
                # y=[total_data.iloc[j] for j in sig_peaks ],
                mode='markers+text',
                text="",
                # text=[str(i) for i in peaks_x],
                name='low',
                textposition='top',
                textfont=dict(
                    size=15
                ),
                marker=dict(
                    size=5,
                    color='rgb(100,0,0)',
                    symbol=1
                ),
            )
            all_data.append(data)
            all_data.append(trace)
            all_data.append(trace_min)
            all_data.append(trace_area)
            all_name.append(name)
        #subplot
        """
        fig = tools.make_subplots(rows=2, cols=2, subplot_titles=(u'機械処理(大王)',u'機械処理(SUGINO)','Reference', 'TEMPO酸化'),)
        xname='time'
        yname='count'
        autorange = 'autorange'
        fig['layout'] ['xaxis1'].update(title=xname,autorange=autorange)
        fig['layout'] ['xaxis2'].update(title=xname,autorange=autorange)
        fig['layout'] ['xaxis3'].update(title=xname,autorange=autorange)
        fig['layout'] ['xaxis4'].update(title=xname,autorange=autorange)
        fig['layout'] ['yaxis1'].update(title=yname)
        fig['layout'] ['yaxis2'].update(title=yname)
        fig['layout'] ['yaxis3'].update(title=yname)
        fig['layout'] ['yaxis4'].update(title=yname)
        for i in np.arange(0,4,1):
            fig['layout']['annotations'][i]['font'].update(size=20)
        fig['layout'].update(height=700, width=900, title=title + ' Charts')
        
        for n,trace in enumerate(all_data):
            if n < 3:
                fig.append_trace(trace, 1, 1)
            elif n < 7 :
                fig.append_trace(trace, 1, 2)
            elif n < 9 :
                fig.append_trace(trace, 2, 1)
            elif n < 11:
                fig.append_trace(trace, 2, 2)
        """
        #all plot
        sep_data = []
        for n,data  in enumerate(all_data):
            if n == 0:
                #pass
                sep_data.append(data)
            elif n ==1:
                #pass
                sep_data.append(data)
            elif n == 8:
                #pass
                sep_data.append(data)
            elif n==9:
                #pass
                sep_data.append(data)
            else:
                sep_data.append(data)
        #fig = dict(data=sep_data, layout=layout)

        for sep_result,sep_name in zip(all_data, all_name):
            if (num == None):
                title = sep_name.replace(".","") + " total_Ion"
            else:
                title = sep_name.replace(".","") + " mz " + str(num)
            layout = dict(
                title = title,
                xaxis=dict(title=self.xname, autorange=self.autorange),
                yaxis=dict(title=self.yname),
                legend=dict(
                    x=0.7,
                    y=1,
                    #orientation="h",
                    traceorder='normal',
                    font=dict(
                        family='sans-serif',
                        size=20,
                        color='#000'
                    ),
                    # bgcolor='#E2E2E2',
                    # bordercolor='#FFFFFF',
                    # borderwidth=2
                )
            )
           # fig = dict(data=1[sep_result], layout=layout)
            #fig = dict(data=sep_data, layout=layout)
            #fileName =title + '.html'
            #plot(fig, filename= fileName,show_link=False)
        #全体保存
        resultDf.columns = all_name
        finalName = DIR + "\\area\\mz_" +num + "_AreaResult.csv"
        #resultDf.to_csv(finalName)
        return resultDf
if __name__ == '__main__':
    plot2D=For2Dplot()
    DIR = 'C:\\googledrive\\export\\for2D'
    base ="C:\\googledrive\\export\\for2D\\base\\kara__1_pivot.csv"
    #DIR = "C:\\googledrive\\export\\for2D\\base"
    #plot2D.read_csv2visualive(DIR,base)
    conectDf = pd.DataFrame()
    for n in np.arange(45,200,1):
        print(num)
        result = plot2D.read_csv2visualive(DIR,None,n)
        conectDf = pd.concat([conectDf,result])
    finalName = DIR + "\\area\\" + "all_mz.csv"
    conectDf.to_csv(finalName)
