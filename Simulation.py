import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time


# The Simulation should contain
# 1.Dataframe (self.snapshots)
# 2.Dictionary (self.connections)
# 3.Settings (self.setngs)
# The dictionary should contain  the states of the persons.
# The columns names would be from number that can be transformed to the position
# and each row would be the state of the grid that particular day.

# Class Settings used to pass settings
class Settings:
    def __init__(self, m=40, n=25, r=6, k=20, a_inf=0.01, a_recov=0, b_recov=0.05, b_death=0.005, g=0.075, N=100):
        self.m = m
        self.n = n
        self.r = r
        self.k = k
        self.a_inf = a_inf
        self.a_recov = a_recov
        self.b_recov = b_recov
        self.b_death = b_death
        self.g = g
        self.N = N


class Simulation:
    # contains the local variables
    # self.setngs - keeps the settings
    # self.snapshots - a data frame that each row is a snapshot of the current situtation
    # self.connections - a dictionary of numbered items to lists that hold the connections of each item

    # Constructor
    def __init__(self, settings):
        d = {}
        self.setngs = settings
        # create tuple
        for i in range(self.setngs.m * self.setngs.n):
            d[i] = self.__get_state()
        self.snapshots = pd.DataFrame(d)
        # create connection dictionary
        self.connections = {}
        self.__set_connections()

    # Runs the simulation based on the number specified in settings
    def run(self):
        for i in range(self.setngs.N):
            self.__helper_run()

    # plot state
    def plot_state(self, Number):
        # prepare plot
        plt.figure(figsize=(60, 60))
        plt.axis("off")
        # draw connections
        for key, value in self.connections.items():
            x1_t = self.__get_position(key)
            x = [x1_t[0]]
            y = [x1_t[1]]
            for i in value:
                x2_t = self.__get_position(i)
                x.append(x2_t[0])
                y.append(x2_t[1])
                plt.plot(x, y, color="g")
                x.append(x1_t[0])
                y.append(x1_t[1])
        # draw spots
        row = self.snapshots.iloc[Number, :]  # I Assume dates start from 0
        for index, value in row.items():
            if value == "S":
                item = list(self.__get_position(index))
                plt.plot(item[0], item[1], marker="o", color="y")
            elif value == "R":
                item = list(self.__get_position(index))
                plt.plot(item[0], item[1], marker="o", color="b")
            elif value == "D":
                item = list(self.__get_position(index))
                plt.plot(item[0], item[1], marker="o", color="k")
            elif value == "I":
                item = list(self.__get_position(index))
                plt.plot(item[0], item[1], marker="o", color="r")
        plt.show()

    def chart(self):
        susceptible = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "S"), axis=1))
        susceptible = susceptible.reset_index(drop=True)
        susceptible = susceptible.rename(columns={0: "susceptible"})

        infected = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "I"), axis=1))
        infected = infected.reset_index(drop=True)
        infected = infected.rename(columns={0: "infected"})

        recovered = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "R"), axis=1))
        recovered = recovered.reset_index(drop=True)
        recovered = recovered.rename(columns={0: "recovered"})

        dead = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "D"), axis=1))
        dead = dead.reset_index(drop=True)
        dead = dead.rename(columns={0: "dead"})

        plt.figure()
        plt.plot(susceptible["susceptible"], color="y")
        plt.plot(infected["infected"], color="r")
        plt.plot(recovered["recovered"], color="b")
        plt.plot(dead["dead"], color="k")
        plt.legend(labels=["susceptible", "infected", "recovered", "dead"])
        plt.show()

    def max_infected(self):
        infected = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "I"), axis=1))
        infected = infected.reset_index(drop=True)
        infected = infected.rename(columns={0: "infected"})
        return infected["infected"].max()

    def peak_infected(self):
        infected = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "I"), axis=1))
        infected = infected.reset_index(drop=True)
        infected = infected.rename(columns={0: "infected"})
        return infected["infected"].idxmax()

    @classmethod
    def averaged_chart(cls, settings, NumberOfSimulations):
        settngs = Settings()
        g = Simulation(settngs)
        g.run()
        df = g.__get_df()
        max_infected = g.max_infected()
        peak_infected = g.peak_infected()

        for i in range(NumberOfSimulations - 1):
            g2 = Simulation(settngs)
            g2.run()
            df2 = g2.__get_df()
            max_infected += g2.max_infected()
            peak_infected += g2.peak_infected()
            df = df.add(df2)
        df = df / NumberOfSimulations

        plt.figure()
        plt.plot(df["susceptible"], color="y")
        plt.plot(df["infected"], color="r")
        plt.plot(df["recovered"], color="b")
        plt.plot(df["dead"], color="k")
        plt.legend(labels=["susceptible", "infected", "recovered", "dead"])
        plt.show()
        max_infected = max_infected / NumberOfSimulations
        peak_infected = peak_infected / NumberOfSimulations

        # I wasnt sure if you wanted this function to return the values as tuples or print them so I did both
        print(f"Max infected average is {max_infected} and the peak infected average is {peak_infected}")
        return (max_infected, peak_infected)

    ########################################################################################################################
    # private functions used by the above functions

    ########################################################################################################################

    # This function simulate one day passed
    def __helper_run(self):
        # add a row of the column names to the end of snapshots
        # the purpose for this is to know inside __are_contacts_infected
        # which column we are at the time being. (Really ugly solution but it works)
        temp = pd.DataFrame(self.snapshots.columns)
        temp = temp.transpose()
        self.snapshots = self.snapshots.append(temp)
        # apply function to row
        self.snapshots.iloc[-1] = self.snapshots.iloc[-1].apply(self.__function_2apply)

    # This function gives the statistics per day of the simulation
    def __get_df(self):
        susceptible = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "S"), axis=1))
        susceptible = susceptible.reset_index(drop=True)
        susceptible = susceptible.rename(columns={0: "susceptible"})

        infected = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "I"), axis=1))
        infected = infected.reset_index(drop=True)
        infected = infected.rename(columns={0: "infected"})

        recovered = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "R"), axis=1))
        recovered = recovered.reset_index(drop=True)
        recovered = recovered.rename(columns={0: "recovered"})

        dead = pd.DataFrame(self.snapshots.apply(lambda row: sum(row[:] == "D"), axis=1))
        dead = dead.reset_index(drop=True)
        dead = dead.rename(columns={0: "dead"})

        df = pd.concat([susceptible, infected, recovered, dead], axis=1)
        return df

    def __get_state(self):
        return np.random.choice(["I", "R", "S"], 1, True,
                                [self.setngs.a_inf,
                                 self.setngs.a_recov,
                                 1 - self.setngs.a_inf - self.setngs.a_recov])

    # transform the column to position x,y
    def __get_position(self, number):
        temp = (number // self.setngs.n, number % self.setngs.n)
        return temp

    # initialize connections
    def __set_connections(self):
        con_counter = 0
        while con_counter <= self.setngs.m * self.setngs.n * self.setngs.k / 2:
            li = random.sample(self.snapshots.columns.tolist(), 2)
            if self.__distance(li) <= self.setngs.r:
                if not self.__are_connected(li):
                    # connect them
                    self.__update_connections(li)
                    con_counter += 1

    def __update_connections(self, li):
        if li[0] in self.connections:
            self.connections[li[0]].append(li[1])
        else:
            self.connections[li[0]] = []
            self.connections[li[0]].append(li[1])
        if li[1] in self.connections:
            self.connections[li[1]].append(li[0])
        else:
            self.connections[li[1]] = []
            self.connections[li[1]].append(li[0])

    def __distance(self, li):
        x1 = self.__get_position(li[0])
        x2 = self.__get_position(li[1])
        temp = math.sqrt(((x1[0] - x2[0]) ** 2) + (x1[1] - x2[1]) ** 2)
        return temp

    def __are_connected(self, li):
        if li[0] in self.connections.keys():
            if li[1] in self.connections[li[0]]:
                return True
            else:
                return False
        else:
            return False

    def __are_contacts_infected(self, number):
        try:
            y = list(map(lambda x: True if self.snapshots.iat[-2, x] == "I" else False, self.connections[number]))
        except KeyError as e:
            return False
        if True in y:
            return True
        else:
            return False

    def __function_2apply(self, position_c):
        if self.snapshots.iat[-2, position_c] == "S":
            if self.__are_contacts_infected(position_c):
                return np.random.choice(["I", "S"], 1, True,
                                        [self.setngs.g,
                                         1 - self.setngs.g,
                                         ])[0]
            else:
                return "S"
        elif self.snapshots.iat[-2, position_c] == "R":
            return "R"
        elif self.snapshots.iat[-2, position_c] == "D":
            return "D"
        elif self.snapshots.iat[-2, position_c] == "I":
            return np.random.choice(["I", "D", "R"], 1, True,
                                    [1 - self.setngs.b_death - self.setngs.b_recov,
                                     self.setngs.b_death,
                                     self.setngs.b_recov])[0]


s = Settings(m=40, n=25, r=2, k=4, N=150)
sim = Simulation(s)
sim.run()
sim.plot_state(0)  # assumed first day is 0
sim.plot_state(99)  # this is 100 day
sim.chart()
print(f"For the first model the max infected were {sim.max_infected()}")
print(f"For the first model the peak infected were {sim.peak_infected()}")
start = time.time()
print(Simulation.averaged_chart(s, 20))
end = time.time()
print(end - start)
