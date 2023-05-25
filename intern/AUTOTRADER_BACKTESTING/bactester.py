from Trading import Trader

at = Trader()

data = input("Write the choice to take data (yahoo/mongo/broker/csv) :- ")
plt = input("Write the choice to take plot or not (1/0) :- ")
strt = input("Write the starting date in (dd/mm/yyyy) :- ")
end = input("Write the end date in (dd/mm/yyyy) :- ")
streg = input("Write the Stretegy to be deployed(ema_crossover) :- ")
intval = input("Write the Amount :- ")
leveage = input("Write the Amount to borrow if needed :- ")
# data  = "yahoo"
# plt = 0
# strt = "22/05/2023"
# end = "25/05/2023"
# streg = "ema_crossover"
# intval = "1000"
# leveage = "100"
at.configure(verbosity=1, show_plot=True, feed=str(data))

at.add_strategy("ema_crossover")

at.Backtesting(start = '1/1/2017', end = '1/5/2022')

at.virtual_account_config(initial_balance=int(intval), leverage = int(leveage))

at.run()





