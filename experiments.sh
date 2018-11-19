# Room environment, MCEIRL vs. sampling vs. baselines
pythonw test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 7
pythonw test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 7
pythonw test.py -e room -p default -c add_rewards -i deviation -d true_reward,final_reward -H 7 -w 0.5
pythonw test.py -e room -p default -c add_rewards -i reachability -d true_reward,final_reward -H 7

# Room environment, same thing, but using a prior
pythonw test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 5
pythonw test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 5

# Room environment, same thing, but uniform prior over initial states
pythonw test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 5 -u True
pythonw test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 5 -u True


# Vases environment, MCEIRL vs. sampling vs. baselines
# Currently doesn't work right, need to debug
# pythonw test.py -e vases -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4
# pythonw test.py -e vases -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4
# pythonw test.py -e vases -p default -c add_rewards -i deviation -d true_reward,final_reward -s 0,1,2,3,4
# pythonw test.py -e vases -p default -c add_rewards -i reachability -d true_reward,final_reward -s 0,1,2,3,4

# Room with train environment
pythonw test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 8
pythonw test.py -e train -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 8
pythonw test.py -e train -p default -c add_rewards -i deviation -d true_reward,final_reward -H 8 -w 0.5
pythonw test.py -e train -p default -c add_rewards -i reachability -d true_reward,final_reward -H 8

# Apples environment, MCEIRL vs. sampling vs. baselines. Reachability won't work.
pythonw test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 11
pythonw test.py -e apples -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 11
pythonw test.py -e apples -p default -c add_rewards -i deviation -d true_reward,final_reward -H 11 -w 0.5
pythonw test.py -e apples -p default -c add_rewards -i reachability -d true_reward,final_reward -H 11

# Long horizon apples?
pythonw test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 20
pythonw test.py -e apples -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 20

# Bad room environment
pythonw test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 5
pythonw test.py -e room -p bad -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 5
pythonw test.py -e room -p bad -c add_rewards -i deviation -d true_reward,final_reward -H 5 -w 0.5
pythonw test.py -e room -p bad -c add_rewards -i reachability -d true_reward,final_reward -H 5

# Batteries
pythonw test.py -e batteries -c add_rewards -d true_reward,final_reward -s 0 -H 11 -i mceirl -p easy
