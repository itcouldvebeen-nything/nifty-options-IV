import re
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import HuberRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATASET_FILE  = "dataset.csv"
SANDBOX_FILE  = "sandbox_solution.csv"
OUTPUT_FILE   = "FINAL_SUBMISSION_ZERO_LOOKAHEAD.csv"

# Ultra-low variance hyperparameters
BLEND = 0.08          
K_NN  = 40            
SEEDS = [42, 123, 456, 789, 2024, 888, 999, 111, 222, 333] # 10-Seed Mega Ensemble
EXPIRY_DT = pd.Timestamp("2026-01-27 15:30:00") 
SHRINK_LAMBDA = 0.001
IDW_SMOOTH = 50
QUAD_ALPHA = 0.015
TEMP_BETA = 0.01
SURFACE_ALPHA = 0.02
TIME_WINDOW = 6 

def parse_instr(n): 
    m = re.match(r"NIFTY(\d+[A-Z]+\d{2})(\d+)(CE|PE)", n)
    if m: 
        return int(m.group(2)), m.group(3)
    m2 = re.search(r'(\d+\.?\d*)', str(n))
    if m2: 
        return int(float(m2.group(1))), 'CE'
    return None, None

def lin(s, v, t): 
    return float(interp1d(s, v, kind='linear', fill_value='extrapolate')(t))

def main():
    print("Loading data and initializing environment...")
    dataset = pd.read_csv(DATASET_FILE)
    sandbox = pd.read_csv(SANDBOX_FILE)
    
    dataset["ts_dt"] = pd.to_datetime(dataset["datetime"], format="%d-%m-%Y %H:%M")
    dataset["T"] = dataset["ts_dt"].apply(lambda t: max((EXPIRY_DT-t).total_seconds()/(365.25*24*3600), 1e-6))
    dataset = dataset.sort_values("ts_dt").reset_index(drop=True)
    
    ic, sm, tm = [], {}, {}
    for c in dataset.columns:
        strike, opt_type = parse_instr(c)
        if strike is not None:
            ic.append(c)
            sm[c] = strike
            tm[c] = opt_type

    N = len(dataset)
    iv_arr = dataset[ic].values
    
    spot_col = next((c for c in ["underlying_price", "underlying", "underlying_value"] if c in dataset.columns), ic[0])
    spot_arr = dataset[spot_col].values
    
    T_arr = dataset["T"].values
    strikes_arr = np.array([sm[c] for c in ic])
    types_arr = np.array([1 if tm[c] == 'CE' else 0 for c in ic])
    ts_strings = dataset["datetime"].values
    
    instr_obs = {ci: [(i, iv_arr[i,ci]) for i in range(N) if not np.isnan(iv_arr[i,ci])] for ci in range(len(ic))}
    obs_2d = {"CE": [], "PE": []}
    
    for ri in range(N):
        for ci in range(len(ic)):
            iv = iv_arr[ri, ci]
            if not np.isnan(iv):
                obs_2d['CE' if types_arr[ci] == 1 else 'PE'].append((ri, strikes_arr[ci], iv))
    for h in obs_2d: 
        obs_2d[h] = np.array(obs_2d[h])

    def base_pred(ri, ci, exc):
        strike = strikes_arr[ci]
        is_ce = types_arr[ci]
        spot = spot_arr[ri]
        ht = 'CE' if is_ce == 1 else 'PE'
        
        tmk = (types_arr == is_ce).copy()
        if exc: 
            tmk[ci] = False
            
        ivt = iv_arr[ri, tmk]
        stt = strikes_arr[tmk]
        ok = ~np.isnan(ivt)
        
        if ok.sum() < 2: 
            return 0.15
        
        s = stt[ok]
        v = ivt[ok]
        si = np.argsort(s)
        s, v = s[si], v[si]
        p = lin(s, v, strike)
        
        if s[0] <= strike <= s[-1] and len(s) >= 5:
            k = min(len(s), 7)
            dd = np.abs(s - strike)
            nb = np.argsort(dd)[:k]
            sn, vn = s[nb], v[nb]
            si2 = np.argsort(sn)
            if len(sn) >= 3: 
                p = (1 - QUAD_ALPHA) * p + QUAD_ALPHA * float(np.polyval(np.polyfit(sn[si2], vn[si2], 2), strike))
            
        dists = np.abs(s - strike)
        weights = 1.0 / (dists + IDW_SMOOTH)
        p = (1 - SHRINK_LAMBDA) * p + SHRINK_LAMBDA * np.average(v, weights=weights)
        
        # STRICT ZERO LOOKAHEAD: Only looking at rows strictly before current row (i < ri)
        nr = sorted([(ri - i, iv_) for i, iv_ in instr_obs[ci] if i < ri])
        if nr: 
            p = (1 - TEMP_BETA) * p + TEMP_BETA * nr[0][1]
        
        ao = obs_2d[ht]
        # STRICT ZERO LOOKAHEAD: Time difference must be strictly positive (past data only)
        time_diff = ri - ao[:, 0]
        tmask = (time_diff >= 0) & (time_diff <= TIME_WINDOW)
        
        if exc: 
            ns = ~((ao[:, 0] == ri) & (ao[:, 1] == strike))
            tmask = tmask & ns
        nb2 = ao[tmask]
        
        if len(nb2) >= 6:
            X = nb2[:, :2]
            Y = nb2[:, 2]
            dt = np.abs(X[:, 0] - ri)
            ds = np.abs(X[:, 1] - strike) / 100
            w = 1.0 / (dt + ds + 1.0)
            Xf = np.column_stack([np.ones(len(X)), X[:, 1], X[:, 0]])
            W = np.diag(w)
            try:
                beta = np.linalg.solve(Xf.T @ W @ Xf, Xf.T @ W @ Y)
                ps = beta[0] + beta[1] * strike + beta[2] * ri
                if np.isfinite(ps) and ps > 0: 
                    p = (1 - SURFACE_ALPHA) * p + SURFACE_ALPHA * ps
            except: 
                pass
        return max(0.001, p)

    def mf(ri, ci, exc):
        strike = strikes_arr[ci]
        is_ce = types_arr[ci]
        spot = spot_arr[ri]
        T = T_arr[ri]
        
        stm = (types_arr == is_ce).copy()
        if exc: 
            stm[ci] = False
            
        sia = iv_arr[ri][stm]
        ssa = strikes_arr[stm]
        om = ~np.isnan(sia)
        sio = sia[om]
        sso = ssa[om]
        
        if len(sio) >= 2:
            iv_mean = sio.mean()
            iv_std = sio.std() if len(sio) > 1 else 0
            si = np.argsort(sso)
            s_, v_ = sso[si], sio[si]
            d2t = np.abs(s_ - strike)
            ni = np.argsort(d2t)
            
            n1 = v_[ni[0]] if len(ni) >= 1 else iv_mean
            n2 = v_[ni[1]] if len(ni) >= 2 else n1
            n3 = v_[ni[2]] if len(ni) >= 3 else n2
            n1d = d2t[ni[0]] if len(ni) >= 1 else 1000
            
            try: 
                p_lin_v = lin(s_, v_, strike)
            except: 
                p_lin_v = iv_mean
                
            try:
                if len(s_) >= 3: 
                    cf = np.polyfit(s_ - spot, v_, 2)
                    curv, slope = cf[0], cf[1]
                else: 
                    slope = (v_[-1] - v_[0]) / max(s_[-1] - s_[0], 1)
                    curv = 0
            except: 
                slope = 0
                curv = 0
        else:
            iv_mean = 0.15
            iv_std = 0
            n1 = n2 = n3 = iv_mean
            n1d = 1000
            p_lin_v = iv_mean
            slope = 0
            curv = 0
            
        sc = iv_arr[:, ci]
        ot = (~np.isnan(sc)).copy()
        if exc: 
            ot[ri] = False
        ti = np.where(ot)[0]
        
        # STRICT ZERO LOOKAHEAD: Past timestamps only
        past_ti = ti[ti < ri]
        if len(past_ti) > 0:
            d2 = ri - past_ti
            ns = past_ti[np.argsort(d2)]
            prev = sc[ns[0]]
            pd_ = ri - ns[0]
            prev2 = sc[ns[1]] if len(ns) >= 2 else prev
            k = min(5, len(ns))
            tmean = np.mean(sc[ns[:k]])
        else:
            prev = iv_mean
            pd_ = 1000
            prev2 = iv_mean
            tmean = iv_mean
            
        log_mon = np.log(strike / spot)
        log_mon_sq = log_mon ** 2 # Mathematical smile injection
            
        return [strike, is_ce, log_mon, log_mon_sq, spot, T, ri/N,
                iv_mean, iv_std, p_lin_v, slope, curv,
                n1, n2, n3, n1d, prev, pd_, prev2, tmean,
                strike - spot, abs(strike - spot)]

    print("Extracting features and training 10-Seed Meta Ensemble...")
    X_tr = []
    y_tr = []
    ri_ci_obs = []
    
    for ri in range(N):
        for ci in range(len(ic)):
            if not np.isnan(iv_arr[ri, ci]):
                X_tr.append(mf(ri, ci, True))
                y_tr.append(iv_arr[ri, ci])
                ri_ci_obs.append((ri, ci))
                
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)

    ens_obs = np.zeros(len(X_tr))
    trained_models = []
    
    # 10 Seeds x 800 Estimators for absolute variance minimization
    for s in SEEDS:
        m = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.025, num_leaves=31,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, random_state=s,
            feature_fraction=0.85, bagging_fraction=0.9, bagging_freq=3, 
            n_jobs=-1, verbose=-1
        )
        m.fit(X_tr, y_tr)
        trained_models.append(m)
        ens_obs += m.predict(X_tr) / len(SEEDS)
        
    base_obs = np.array([base_pred(ri, ci, True) for ri, ci in ri_ci_obs])
    pred_obs = np.maximum((1 - BLEND) * base_obs + BLEND * ens_obs, 0.001)

    print("Generating base predictions for missing values...")
    missing_ri_ci = []
    missing_ids = []
    missing_X = []
    missing_base_preds = []
    
    for ri in range(N):
        ts_s = ts_strings[ri]
        for ci, instr in enumerate(ic):
            if np.isnan(iv_arr[ri, ci]):
                key = f"{ts_s}||{instr}"
                missing_ri_ci.append((ri, ci))
                missing_ids.append(key)
                missing_X.append(mf(ri, ci, False))
                missing_base_preds.append(base_pred(ri, ci, False))
                
    missing_X = np.array(missing_X)
    missing_base_preds = np.array(missing_base_preds)
    
    missing_ens_preds = np.zeros(len(missing_X))
    for m in trained_models:
        missing_ens_preds += m.predict(missing_X) / len(SEEDS)
        
    missing_base = np.maximum((1 - BLEND) * missing_base_preds + BLEND * missing_ens_preds, 0.001)

    print("Running Huber-Clamped Affine Calibrator...")
    ci_to_idx = {ci: [] for ci in range(len(ic))}
    for i, (ri, ci) in enumerate(ri_ci_obs): 
        ci_to_idx[ci].append(i)
        
    calib_a = np.zeros(len(ic))
    calib_b = np.ones(len(ic))
    
    for ci in range(len(ic)):
        idxs = ci_to_idx[ci]
        if len(idxs) < 30: 
            continue
        x = pred_obs[idxs]
        y = y_tr[idxs]
        
        res0 = y - x
        keep = np.abs(res0 - np.median(res0)) < 4.5 * np.std(res0)
        if keep.sum() < 30: 
            continue
            
        x_clean, y_clean = x[keep].reshape(-1, 1), y[keep]
        
        # Robust Huber regression to ignore extreme outliers in deep OTM options
        huber = HuberRegressor(epsilon=1.35)
        huber.fit(x_clean, y_clean)
        
        a, b = huber.intercept_, huber.coef_[0]
        
        # Absolute clamping to prevent exploding slopes on edge strikes
        b = np.clip(b, 0.85, 1.15)
        
        calib_a[ci] = a
        calib_b[ci] = b
        
    calibrated_miss = np.array([calib_a[c] + calib_b[c] * missing_base[i] for i, (r, c) in enumerate(missing_ri_ci)])
    calibrated_obs = np.array([calib_a[c] + calib_b[c] * pred_obs[i] for i, (r, c) in enumerate(ri_ci_obs)])

    print("Applying structural kNN Surface Retrieval...")
    def kf(ri, ci):
        spot = spot_arr[ri]
        strike = strikes_arr[ci]
        # Emphasis on structural parameters over time to ensure cross-sectional safety
        return np.array([np.log(strike/spot) * 3.0, 
                         ri/N, 
                         float(types_arr[ci]) * 0.5,
                         abs(strike-spot)/spot * 2.0, 
                         spot/spot_arr.mean()])
                         
    af = np.array([kf(ri, ci) for ri, ci in ri_ci_obs + missing_ri_ci])
    fs = af.std(axis=0)
    fs[fs < 1e-9] = 1.0
    nf = af / fs
    
    of = nf[:len(ri_ci_obs)]
    mff = nf[len(ri_ci_obs):]
    
    nn = NearestNeighbors(n_neighbors=K_NN, algorithm='ball_tree')
    nn.fit(of)
    d_te, i_te = nn.kneighbors(mff)
    bw = np.median(d_te[:, 1])
    
    w_te = np.exp(-(d_te / bw)**2)
    w_te /= w_te.sum(axis=1, keepdims=True)
    nn_v_te = (w_te * y_tr[i_te]).sum(axis=1)
    
    nn_o = NearestNeighbors(n_neighbors=K_NN+1, algorithm='ball_tree')
    nn_o.fit(of)
    d_oo, i_oo = nn_o.kneighbors(of)
    d_oo = d_oo[:, 1:]
    i_oo = i_oo[:, 1:]
    
    w_oo = np.exp(-(d_oo / bw)**2)
    w_oo /= w_oo.sum(axis=1, keepdims=True)
    nn_v_obs = (w_oo * y_tr[i_oo]).sum(axis=1)

    print("Running grid search for ensemble weights...")
    best_mse = ((y_tr - pred_obs)**2).mean()
    best_a = 0.0
    best_g = 0.0
    
    for a in np.arange(0.0, 1.01, 0.1):
        for g in np.arange(0.0, 1.01 - a + 1e-9, 0.1):
            mse = ((y_tr - ((1 - a - g) * pred_obs + a * calibrated_obs + g * nn_v_obs))**2).mean()
            if mse < best_mse: 
                best_mse = mse
                best_a = a
                best_g = g
            
    final_te = np.clip((1 - best_a - best_g) * missing_base + best_a * calibrated_miss + best_g * nn_v_te, 0.001, None)
    preds = dict(zip(missing_ids, final_te))
    
    rows = [{"id": r["id"], "value": preds.get(r["id"], r["value"])} for _, r in sandbox.iterrows()]
    sub = pd.DataFrame(rows)
    sub.to_csv(OUTPUT_FILE, index=False, float_format="%.16f")
    
    print(f"Submission saved to {OUTPUT_FILE} (alpha={best_a:.2f}, gamma={best_g:.2f})")

if __name__ == "__main__": 
    main()