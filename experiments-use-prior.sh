# ROOM default
python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior

python test.py -e room -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior
python test.py -e room -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior


# ROOM bad
python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior

python test.py -e room -p bad -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior
python test.py -e room -p bad -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior


# TRAIN default
python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.0625 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.125 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.25 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 0.5 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 1 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 2 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 4 -o results/use_prior

python test.py -e train -p default -c use_prior -i mceirl -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior
python test.py -e train -p default -c use_prior -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -H 10 -k 8 -o results/use_prior
