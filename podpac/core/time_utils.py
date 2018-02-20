from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np


def get_time_unit(time):
    unit = time.dtype.name
    unit = unit[unit.index('[') + 1: unit.index(']')]
    attr = {'Y': 'year', 'M': 'month'}[unit]
    return attr


def add_time_coords(base, delta):
    unit = get_time_unit(delta)

    date = base.astype(object)
    delta = np.atleast_1d(delta)

    dates = []
    for d in delta:
        try: 
            dates.append(date.replace(**{unit: getattr(date, unit) + d.astype(object)}))
        except ValueError as e:
            dates.append(date.replace(**{unit: getattr(date, unit) + d.astype(object)
                    , 'day': 28}))
    dates = np.array(dates).astype(np.datetime64)
    if dates.size == 1:
        dates = dates[0]
    return dates


def divide_time_coords(base, delta):
    unit = get_time_unit(delta)
    date = base.astype(object)
    
    try: 
        date = date.replace(**{unit: getattr(date, unit) / delta.astype(object)})
    except ValueError as e:
        date = date.replace(**{unit: getattr(date, unit) / delta.astype(object)
                , 'day': 28})
    return np.datetime64(date)


def get_time_coords_size(start, stop, delta):
    unit = get_time_unit(delta)
    start_date = start.astype(object)
    stop_date = stop.astype(object)
    
    size = (getattr(stop_date, unit) - getattr(start_date, unit)) / delta.item() 
    return int(size)


