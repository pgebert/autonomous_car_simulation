import os.path
import time
import json
import sys

import plotly as py
import plotly.graph_objs as go


def plot(data=None, out_path=None):

    # Set loading flag
    load_flag = False 
    if data == None:
        load_flag = True

    # Set output path
    if out_path == None:
        out_path = './plot.png'


    # Load data from json if necessary
    path = 'log.json'

    if load_flag:

        print("--- Loading: " + path + " ---")
        if os.path.isfile(path) and os.stat(path).st_size != 0:

            jsonFile = open(path, "r+")
            data = json.load(jsonFile)
            jsonFile.close()
        else:
            raise ValueError("%s isn't a file!" % path)


    plotData = []    

    # Loop over different configurations
    for i in range(len(data)):
        x_loss, y_loss = [], []
        for val in data[i]["loss"]:
                x_loss.append(val[0])
                y_loss.append(val[1])
        # Plot loss
        plotData.append(go.Scatter(
        x = x_loss,
        y = y_loss,
        # mode = 'lines',
        mode = 'lines+markers',
        # mode = 'markers',
        name = data[i]["cfg"] + " - Train"
        ))

        x_test, y_test = [], []
        for val in data[i]["test"]:
                x_test.append(val[0])
                y_test.append(val[1])        
        # Plot test
        plotData.append(go.Scatter(
        x = x_test,
        y = y_test,
        # mode = 'lines',
        mode = 'lines+markers',
        # mode = 'markers',
        name = data[i]["cfg"] + " - Test"
        ))

    # py.offline.plot(plotData, auto_open=False)
    # py.offline.plot(plotData, image='png', auto_open=False, filename='./plot.html')    
    py.offline.plot(plotData, auto_open=False, filename='./plot.html')   

    if load_flag:
        print("--- Saved: plot.html ---")
    else:
        sys.stdout.write("Updated plot.html          \r")
        sys.stdout.flush()

if  __name__ =='__main__':
    plot()
