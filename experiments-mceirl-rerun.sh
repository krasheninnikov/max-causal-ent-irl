################################################################################
##########################            ADDING, test w          ##################
################################################################################


# ROOM mceirl
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.0625 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.125 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.25 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.5 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 1 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 2 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 4 -o results/tuning
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 8 -o results/tuning

# ROOM bad mceirl
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.0625 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.125 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.25 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.5 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 1 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 2 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 4 -o results/tuning
python test.py -e room -p bad -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 8 -o results/tuning


# Train mceirl
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.0625 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.125 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.25 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.5 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 1 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 2 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 4 -o results/tuning
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 8 -o results/tuning


# Apples mceirl
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.0625 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.125 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.25 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 0.5 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 1 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 2 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 4 -o results/tuning
python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -w 8 -o results/tuning

################################################################################
##########################            PRIOR, test k      #######################
################################################################################
# ROOM mceirl
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/tuning
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/tuning

# ROOM bad mceirl
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/tuning
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/tuning


# TRAIN mceirl
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/tuning
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/tuning


# Apples mceirl
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/tuning
python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/tuning
