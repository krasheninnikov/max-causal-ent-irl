# ADD REWARDS
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
python test.py -e room -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r


python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
python test.py -e train -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r


# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c add_rewards -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r


# USE PRIOR
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r


python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r


# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
# python test.py -e apples -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r



# # ROOM sampling
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.0625 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.125 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.25 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 0.5 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 1 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 2 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 4 -o results/use_prior_vs_add_r
# python test.py -e room -p default -c add_rewards -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -l 0.003 -m 5000 -H 10 -k 8 -o results/use_prior_vs_add_r
