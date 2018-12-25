'''
This script is used to fit the distribution of ontime, offtime, flow duration, and flow inter-arrival time. In addition, we also caculate the mean data rate in the ON period and probability of tcp/udp flows
'''
import pandas as pd
import numpy as np
import os, sys, getopt
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import warnings

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # Distributions to check
    DISTRIBUTIONS = [
        st.weibull_min, st.lognorm, st.expon, st.pareto, st.erlang, st.gamma]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # fit dist to data
        params = distribution.fit(data)
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))
        dist = getattr(st, distribution.name)
        param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:f}'.format(k,v) for k,v in zip(param_names, params)])
        dist_str = '{}({})'.format(distribution.name, param_str)
        label = dist_str + ", sse=%.4f" % (sse)

        # if axis pass in add to plot
        if ax:
            pd.Series(pdf, x).plot(ax=ax, label= label, legend=True)
        # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf

def plotFitting(data, name):
    cdf, x = np.histogram(data, bins=200)
    cumcdf = np.cumsum(cdf)
    cdf = [num/float(len(data)) for num in cumcdf]
    for i in range(len(cdf)):
        if cdf[i] >= 0.95:
            threshold = x[i+1]
            break
    data = [val for val in data if val < threshold]
    data = pd.Series(data)

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=100, density=True, alpha=0.5) #color=plt.rcParams['axes.color_cycle'][1])
    # Save plot limits
    dataYLim = ax.get_ylim()
    dataXLim = ax.get_xlim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_xlim(dataXLim)
    ax.set_title(name)
    ax.set_xlabel(u'Time/s')
    ax.set_ylabel('Probability density')

    # # Make PDF with best params
    # pdf = make_pdf(best_dist, best_fit_params)
    #
    # # Display
    # plt.figure(figsize=(12,8))
    # ax = pdf.plot(lw=2, label='PDF', legend=True)
    # data.plot(kind='hist', bins=100, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    #
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    print name + ":" + dist_str
    #
    # ax.set_ylim(dataYLim)
    # ax.set_xlim(dataXLim)
    # ax.set_title(name + u'. with best fit distribution \n' + dist_str)
    # ax.set_xlabel(u'Time/s')
    # ax.set_ylabel('Frequency')

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:a:",["ifile=", 'arrival95='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <timeRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -a <arrival95>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-a", "--arrival95"):
            arrival95 = int(arg) # the unit is in ms

    count = 0
    onTimes = []
    offTimes = []
    flowDuration = []
    flowInterArrival = []
    onTimeRates = []
    tcpFlows = 0
    udpFlows = 0
    arrival95 /= 1000.0

    for chunk in pd.read_csv(input_file, usecols=['Protocol','arrivals', 'pktLens'], chunksize=10000):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            arrivals = entry['arrivals'][1:-1].split(' ')
            arrivals = [float(t) for t in arrivals]
            flowDuration.append(arrivals[-1]-arrivals[0])
            flowInterArrival.append(arrivals[0])
            if entry['Protocol'] == 'TCP':
                tcpFlows += 1
            else:
                udpFlows += 1

            interArrivals = [arrivals[i+1]-arrivals[i] for i in range(len(arrivals)-1)]
            pkts = entry['pktLens'][1:-1].split(' ')
            pkts = [int(p) for p in pkts]
            ontime = 0
            offtime = 0
            onPkts = 0
            for i in range(len(interArrivals)):
                if interArrivals[i] <= arrival95:
                    ontime += interArrivals[i]
                    onPkts += pkts[i]
                    if offtime:
                        offTimes.append(offtime)
                    offtime = 0
                else:
                    offtime += interArrivals[i]
                    if ontime:
                        onTimes.append(ontime)
                        onTimeRates.append(onPkts+pkts[i])
                    onPkts = 0
                    ontime = 0

    plotFitting(onTimes, "Length of ON Periods")
    plotFitting(offTimes, "Length of OFF Periods")
    plotFitting(flowDuration, "Length of Flows")
    flowInterArrival.sort()
    flowInterArrival = [flowInterArrival[i+1]-flowInterArrival[i] for i in range(len(flowInterArrival)-1)]
    plotFitting(flowInterArrival, "Flow Inter-Arrival Times")
    onTimeRates = [onTimeRates[i]/onTimes[i] for i in range(len(onTimeRates))]
    print "average data rate in On Period = %f" % np.mean(onTimeRates)
    print "probability to be tcp flows = %f" % (tcpFlows/(tcpFlows + udpFlows + 0.0))
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
