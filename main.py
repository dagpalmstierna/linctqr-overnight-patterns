
from trade_condition import MondayCondition, ReversalCondition, TwoDayDown
from securities import Security

# Initialize security with specific trade conditions
trade_conditions = [ReversalCondition(), TwoDayDown()]


ihyg = Security("IHYG.L", trade_conditions=trade_conditions)
sybj = Security("SYBJ.DE", trade_conditions)
ibcx = Security("IBCX.L", trade_conditions)
stoxx = Security("^STOXX", trade_conditions)
hyg = Security("HYG", trade_conditions)

ihyg.find_reversals()
print(ihyg.get_performance_summary())

#sybj.find_reversals()
#print(sybj.get_performance_summary())

#ibcx.find_reversals()
#print(ibcx.get_performance_summary())

#stoxx.find_reversals()
#print(stoxx.get_performance_summary())

#hyg.find_reversals()
#print(hyg.get_performance_summary())



# Find reversals and print performance summary
