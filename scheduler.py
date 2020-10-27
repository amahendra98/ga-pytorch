import numpy as np

' My Implementation of rate schedulers for training '

class base_scheduler(object):
    ''' Base Class scheduler from which all other schedulers inherit
            schedule: holds the chronological schedule of next values
            current: holds the current value as scheduled
            prior: holds the history of values reached over each epoch or call to scheduler '''
    def __init__(self, step_length, schedule):
        self.schedule = schedule
        self.current = self.param_from_schedule(0)
        self.prior = np.empty(step_length)
        self.prior[:] = np.nan

    ' Action of scheduler changing some kind of value in light of schedule and current condition '
    def check(self, val, gen):
        if len(self.schedule) != 1:
            b = np.isnan(self.prior)

            if np.any(b):
                for i in range(len(self.prior)-1,-1,-1):
                    if np.isnan(self.prior[i]):
                        self.prior[i] = val
                        break
            else:
                self.prior[-1] = val
                self.prior = np.roll(self.prior, 1)

            if self.change_condition(gen):
                self.schedule.pop(0)
                self.current = self.param_from_schedule(0)
                self.reset_prior()

        print(self.prior)
        return self.current

    ' Condition that determines whether or not the next value in the schedule is loaded '
    def change_condition(self, gen):
        return True

    ' Function that grabs a specific value from the scheduler structure '
    def param_from_schedule(self, idx):
        return self.schedule[idx]

    ' Function used to reset the history of values stored from past generations '
    def reset_prior(self):
        self.prior[:] = np.nan




class value_based_scheduler(base_scheduler):
    ''' Scheduler will load next value if the history of values filling the prior array have an average percent error
    from their mean that is less than some threshold '''

    def __init__(self, step_length, schedule, thresh):
        super(value_based_scheduler,self).__init__(step_length, schedule)
        self.perr_thresh = thresh

    def change_condition(self, gen):
        u = np.mean(self.vals)
        perr = np.abs((self.vals - u) / u)

        if perr <= self.perr_thresh:
            return True



class step_length_doubling_scheduler(value_based_scheduler):
    ''' Scheduler will load next value and double the size of prior to hold a greater history of values, if the history
    of values filling the prior array have an average percent error from their mean that is less than some threshold '''
    def __init__(self, step_length, schedule, thresh):
        super(step_length_doubling_scheduler, self).__init__(step_length, schedule, thresh)

    def reset_prior(self):
        self.prior = np.empty(2*len(self.prior))
        self.prior = np.nan


class generational_scheduler(base_scheduler):
    ''' Scheduler will load next value at specific generations '''
    def __init__(self, step_length, schedule, thresh=None):
        super(generational_scheduler, self).__init__(step_length, schedule)

    def param_from_schedule(self, idx):
        return self.schedule[idx][0]

    def change_condition(self, gen):
        return gen >= self.schedule[1][1]