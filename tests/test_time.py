from momi3.MOMI import Momi
from momi3.Params import Params
from demos import FiveDemes, MultiAnc, SingleDeme, ThreeDemes, TwoDemes

SEED = 108
REP = 5

def get_momi():
    g = 0.025
    size = 1000.0
    rate = 0.01
    t = 100.0

    demo, model1 = TwoDemes.Exponential(t=t, size=size, g=g).migration(
        tstart=t, rate=rate
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]

    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    return momi


def get_params(momi):
	params = momi._default_params
	params.set_train_all(True)
	return params


def get_jsfs(momi, ntrees=100):
	jsfs = momi.simulate(ntrees, seed=SEED)
	print('non zero entries:', jsfs.nnz)
	return jsfs


def test_loglik():
	momi = get_momi()
	params = get_params(momi)
	jsfs = get_jsfs(momi)
	val, ct, t = momi._time_loglik(params, jsfs, repeat=REP, average=True)
	print(f'compilation time:{ct}')
	print(f'runtime:{t}')


def test_loglik_with_gradient():
	momi = get_momi()
	params = get_params(momi)
	jsfs = get_jsfs(momi)
	val, ct, t = momi._time_loglik_with_gradient(params, jsfs, repeat=REP, average=True)
	print(f'compilation time:{ct}')
	print(f'runtime:{t}')


if __name__ == "__main__":
	test_loglik()
	#test_loglik_with_gradient()
