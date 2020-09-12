
import numpy as np

def kalman_prediction_old(x, P, F, Q):
    """ 
        Kalman filter prediction for linear motion model.
    """
    if type(x) is int or type(x) is float:
        x_hat = F * x
        P_hat = F * P * F + Q
    else:
        x_hat = F @ x
        P_hat = F @ P @ F.T + Q
    return x_hat, P_hat

def kalman_update_old(x, P, H, R, z):
    """ 
        Kalman filter update for linear measurement model.
    """
    if type(x) is int or type(x) is float:
        z_hat = H * x # Predicted measurement
        epsi = z - z_hat # Innovation
        S = H * P * H + R # Innovation covariance
        K = P * H / S # Kalman gain
        x_hat = x + K * epsi # Weighted average
        P_hat = P - K * H * P # Decrease uncertainty
    else:
        z_hat = H @ x # Predicted measurement
        epsi = z - z_hat # Innovation
        S = H @ P @ H.T + R # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S) # Kalman gain
        x_hat = x + K @ epsi # Weighted average
        P_hat = P - K @ H @ P # Decrease uncertainty
    return x_hat, P_hat

def kalman_filter_old(x_init, P_init, Zs, F, Q, H, R):
    """ 
        Kalman filter for linear motion and measurement models.
    """
    N = len(Zs)
    X_hat = [x_init]
    P_hat = [P_init]
    for k in range(N):
        x_hat_kmin1, P_hat_kmin1 = kalman_prediction_old(X_hat[k], P_hat[k], F, Q)
        x_hat_k, P_hat_k = kalman_update_old(x_hat_kmin1, P_hat_kmin1, H, R, Zs[k])
        X_hat.append(x_hat_k)
        P_hat.append(P_hat_k)
    return X_hat, P_hat

def kalman_prediction(x, P, F, Q):
    """ 
        Kalman filter prediction for linear motion model.
        x - numpy array with shape (n,)
        P, F, Q - numpy array with shape (n,n)
    """
    x_hat = F @ x
    P_hat = F @ P @ F.T + Q
    return x_hat, P_hat

def kalman_update(x, P, H, R, z):
    """ 
        Kalman filter update for linear measurement model.
        x - numpy array with shape (n,)
        P - numpy array with shape (n,n)
        H - numpy array with shape (n,m)
        R - numpy array with shape (m,m)
        z - numpy array with shape (m,)
    """
    z_hat = H @ x # Predicted measurement
    epsi = z - z_hat # Innovation
    S = H @ P @ H.T + R # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S) # Kalman gain
    x_hat = x + K @ epsi # Weighted average
    P_hat = P - K @ H @ P # Decrease uncertainty
    return x_hat, P_hat

def kalman_filter(x_init, P_init, Zs, F, Q, H, R):
    """ 
        Kalman filter for linear motion and measurement models.
    """
    N = len(Zs)
    nk = len(x_init)
    X_hat = np.zeros((nk,N+1))
    P_hat = np.zeros((nk,nk,N+1))
    X_hat[:,0] = x_init
    P_hat[:,:,0] = P_init
    for k in range(1,N+1):
        x_hat_kmin1, P_hat_kmin1 = kalman_prediction(X_hat[:,k-1], P_hat[:,:,k-1], F, Q)
        X_hat[:,k], P_hat[:,:,k] = kalman_update(x_hat_kmin1, P_hat_kmin1, H, R, Zs[k-1])
    return X_hat, P_hat


""" Simple kalman filter tests """
def _test_scalar_kalman_filter_old():
    x_init = 0.9
    P_init = 0.2
    Zs = [1, 1.1, 1.1, 1.2]
    F = 1
    Q = 0.3
    H = 1
    R = 0.2
    x_hats, P_hats = kalman_filter_old(x_init, P_init, Zs, F, Q, H, R)
    print(x_hats)

def _test_kalman_filter_old():
    ones = np.ones((2,1))
    I = np.eye(2)

    x_init = 0.9 * ones
    P_init = 0.3 * I
    Zs = [ones, 1.1*ones, 1.1*ones, 1.2*ones]
    F = I
    Q = 0.3 * I
    H = I
    R = 0.2 * I
    x_hats, P_hats = kalman_filter_old(x_init, P_init, Zs, F, Q, H, R)
    print(x_hats)

def _test_scalar_kalman_filter():
    sarr = lambda scalar: np.array([[scalar]])
    x_init = sarr(0.9)
    P_init = sarr(0.2)
    Zs = [1, 1.1, 1.1, 1.2]
    F = sarr(1)
    Q = sarr(0.3)
    H = sarr(1)
    R = sarr(0.2)
    x_hats, P_hats = kalman_filter(x_init, P_init, Zs, F, Q, H, R)
    [print(x) for x in x_hats.T]

def _test_kalman_filter():
    ones = np.ones((2,))
    I = np.eye(2)

    x_init = 0.9 * ones
    P_init = 0.3 * I
    Zs = [ones, 1.1*ones, 1.1*ones, 1.2*ones]
    F = I
    Q = 0.3 * I
    H = I
    R = 0.2 * I
    x_hats, P_hats = kalman_filter(x_init, P_init, Zs, F, Q, H, R)
    [print(x) for x in x_hats.T]

if __name__ == "__main__":
    # _test_scalar_kalman_filter_old()
    # _test_kalman_filter_old()
    _test_scalar_kalman_filter()
    _test_kalman_filter()