#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime
from scipy.interpolate import Rbf

class patternRecognition(object):

    def __init__(self, prices, linear_threshold, outer_slope_threshold, inner_slope_threshold, minimum_range):
        self.prices = prices
        self.linear_threshold = linear_threshold #.03
        self.outer_slope_threshold = outer_slope_threshold #.05
        self.inner_slope_threshold = inner_slope_threshold #.2
        self.minimum_range = minimum_range #.1
        self.window_size = window_size #40

    def standardize(self):
        return (self.prices / self.prices[0]) - 1

    def find_max_min(self, smooth=0.2):
        '''
        Given price window, apply smoothing function, and find relative extrema (max and min)
        ---
        IN:
            prices: np.array, price time series
            smooth: float, smoothing constant
        OUT:
            max_min: pd.DataFrame of relative extrema along with their respective indicies in the original prices series
        '''
        prices = pd.Series(self.prices)
        prices.index = np.linspace(1., len(prices), len(prices))
        rbf = Rbf(prices.index, prices, smooth=smooth)
        smooth_prices = pd.Series(rbf(prices.index), index=prices.index)

        local_max = argrelextrema(smooth_prices.values, np.greater)[0]
        local_min = argrelextrema(smooth_prices.values, np.less)[0]

        # Find maxima
        price_local_max_dt = []
        for i in local_max:
            if i > 1 and i < len(prices) - 2:
                price_local_max_dt.append(prices.iloc[i-1:i+2].idxmax())

        # Find minima
        price_local_min_dt = []
        for i in local_min:
            if i > 1 and i < len(prices) - 2:
                price_local_min_dt.append(prices.iloc[i-1:i+2].idxmax())

        maxima = pd.DataFrame(prices.loc[price_local_max_dt])
        minima = pd.DataFrame(prices.loc[price_local_min_dt])

        max_min = pd.concat([maxima, minima]).sort_index()

        return max_min

    def find_patterns(self):

        '''
        For each 40-day window, apply following function/decision tree to determine if specific patterns exists among extrema, only one pattern may be identified for each window, decisions are made with either spatial or slope based rules, different parameters may be adjusted in order for rules to be more strict/lenient, patterns must have at least 5 extrema and may contain more, if a pattern doesn't complete before the end of the window then it is not added
        ---
        IN:
            max_min: pd.DataFrame, all relative extrema for the window
            linear_threshold: float, threshold for instances that require flat (slope=0) trends
            outer_slope_threshold: float, depending on whether the line is resistance or support, this value is typically lower as we care more about whether a price breaks out or below these outer trend lines
            inner_slope_threshold: float, opposite of outer_slope_threshold, typically larger since extrema occuring within the preexisting trend are acceptable
            minimum_range: float, minimum difference between the max and min of a potential pattern
            window_size: 40

        OUT:
            pattern: string, one of 18 patterns or 'None'
        '''
        max_min = find_max_min()

        spread = float(max_min.max() - max_min.min())
        zero = float(linear_threshold * spread)

        def transform(x):
            return (spread / (window_size - 1)) * (x - 1) + max_min.min()

        for i in range(len(max_min) - 4):
            window = max_min.iloc[i:i+5]
            if float((window.max() - window.min()) / spread) < minimum_range:
                continue

            e1 = float(window.iloc[0])
            e2 = float(window.iloc[1])
            e3 = float(window.iloc[2])
            e4 = float(window.iloc[3])
            e5 = float(window.iloc[4])

            # Head and Shoulders (Bearish Reversal)
            if (e1 > e2) and (e3 > e1) and (e3 > e5) and (e5 > e4) and \
                (abs(e1 - e5) <= zero) and \
                (abs(e2 - e4) <= zero):
                    return 'head_shoulders'
                    #patterns['head_shoulder'].append((window.index[0], window.index[-1]))

            # Inverse Head and Shoulders (Bullish Reversal)
            elif (e1 < e2) and (e3 < e1) and (e3 < e5) and (e5 < e4) and \
                (abs(e1 - e5) <= zero) and \
                (abs(e2 - e4) <= zero):
                    return 'inverse_head_shoulders'

            elif (abs(e1 - e3) <= zero) and \
                (abs(e1 - e5) <= zero) and \
                (abs(e3 - e5) <= zero):

                # Triple Bottom (Bullish Reversal)
                if (e1 < e2) and (e3 < e4) and (e5 < e4) and \
                    (abs(e2 - e4) <= zero):
                        return 'trip_bottom'

                # Triple Top (Bearish Reversal)
                elif (e1 > e2) and (e3 > e4) and (e5 > e4) and \
                    (abs(e2 - e4) <= zero):
                        return 'trip_top'

            else:

                extrema = [e1, e2, e3, e4]
                indicies = [window.index[j] for j in range(4)]

                slope_odds = float((e3 - e1) / (transform(indicies[2]) - transform(indicies[0])))
                slope_evens = float((e4 - e2) / (transform(indicies[3]) - transform(indicies[1])))
                base_slopes = [slope_odds, slope_evens] #[0 = odds, 1 = evens]

                if e1 > e2:
                    direction = 'top'
                else:
                    direction = 'bottom'

                # Loop to add additional extrema
                for k in range(4+i,len(max_min)):
                    e = max_min.iloc[k]
                    index = max_min.index[k]

                    sign = k % 2    # sign = 0 (Odds), sign = 1 (Evens)
                    reference_slope = base_slopes[sign]
                    slope = float((e - extrema[sign]) / (transform(index) - transform(indicies[sign])))

                    lower_bound = None
                    upper_bound = None

                    # Check slopes
                    if abs(reference_slope) < zero and abs(slope) < zero:
                        lower_bound = 'flat'
                        upper_bound = 'flat'

                    elif (direction == 'top' and sign == 0 and reference_slope >= zero) or \
                        (direction == 'bottom' and sign == 1 and reference_slope >= zero):
                            lower_bound = reference_slope*(1-inner_slope_threshold)
                            upper_bound = reference_slope*(1+outer_slope_threshold)

                    elif (direction == 'top' and sign == 0 and reference_slope <= -zero) or \
                        (direction == 'bottom' and sign == 1 and reference_slope <= -zero):
                            lower_bound = reference_slope*(1+inner_slope_threshold)
                            upper_bound = reference_slope*(1-outer_slope_threshold)

                    elif (direction == 'top' and sign == 1 and reference_slope >= zero) or \
                        (direction == 'bottom' and sign == 0 and reference_slope >= zero):
                            lower_bound = reference_slope*(1-outer_slope_threshold)
                            upper_bound = reference_slope*(1+inner_slope_threshold)

                    elif (direction == 'top' and sign == 1 and reference_slope <= -zero) or \
                        (direction == 'bottom' and sign == 0 and reference_slope <= -zero):
                            lower_bound = reference_slope*(1+outer_slope_threshold)
                            upper_bound = reference_slope*(1-inner_slope_threshold)

                    if lower_bound and upper_bound:

                        if lower_bound == 'flat' and upper_bound =='flat':
                            extrema.append(e)
                            indicies.append(index)
                            #print('added')

                        elif lower_bound <= slope and slope <= upper_bound:
                            extrema.append(e)
                            indicies.append(index)
                            #print('added')

                        else:
                            break

                    else:
                        break



                # Must have at least 5 extrema and can't contain the final extrema (a check on whether the pattern is completed)
                if len(extrema) < 5 or max_min.index[-1] in indicies:
                    continue

                # Right Angles (Reversals)
                if abs(slope_evens) <= zero:
                    if direction == 'bottom':
                        if slope_odds >= zero:
                            return 'right_tri_bottom'
                        elif slope_odds <= -zero:
                            return 'right_broad_bottom'
                    else:
                        if slope_odds <= -zero:
                            return 'right_tri_top'
                        elif slope_odds >= zero:
                            return 'right_broad_top'

                # Broadening (Reversals) [Higher highs, lower lows]
                elif slope_odds <= -zero and slope_evens >= zero:
                    if direction == 'bottom':
                        return 'broad_bottom'

                    else:
                        return 'broad_top'

                # Wedges/Triangles (Reversals) [Lower highs, higher lows]
                elif slope_odds >= zero and slope_evens <= -zero:
                    if direction =='bottom':
                        return 'wedge_bottom'

                    else:
                        return 'wedge_top'

                # Continuation triangles
                elif abs(slope_odds) <= zero:
                    if direction == 'bottom':
                        if slope_evens <= -zero:
                            return 'desc_tri'
                    else:
                        if slope_evens >= zero:
                            return 'asc_tri'

                # Channel Up [Higher highs, higher lows]
                elif slope_odds >= zero and slope_evens >= zero:
                    if direction == 'bottom':
                        return 'bearish_channel_up'
                    else:
                        return 'bullish_channel_up'

                # Channel Down [Lower highs, lower lows]
                elif slope_odds <= -zero and slope_evens <= -zero:
                    if direction == 'bottom':
                        return 'bearish_channel_down'
                    else:
                        return 'bullish_channel_down'
