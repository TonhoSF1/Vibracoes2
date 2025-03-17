import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, signal


def Linear_System_MDoF(t, x, M_int_inv, C_int, K_int, F_int):
    """
    Resolve o sistema dinâmico de múltiplos graus de liberdade (MDoF).

    A função implementa a equação do movimento:
        M * x" + C * x' + K * x = F(t)
    onde:
        - x: vetor de deslocamentos (n)
        - x': vetor de velocidades (n)
        - x": vetor de acelerações (n)
        - M: matriz de massa (n x n)
        - C: matriz de amortecimento (n x n)
        - K: matriz de rigidez (n x n)
        - F(t): vetor de forças externas (n)

    O sistema é reescrito como um sistema de equações de primeira ordem:
        y = [x; x']
        dy/dt = [x'; M⁻¹(F(t) - C*x' - K*x)]

    Parâmetros:
    ----------
    t : float
        Tempo atual da simulação (não é usado diretamente na fórmula, mas necessário para integração numérica).
    x : numpy.ndarray
        Vetor de estado concatenado:
            x = [deslocamentos (x); velocidades (x')].
            Dimensão: (2n,), onde n é o número de graus de liberdade (DoFs).
    M_int_inv : numpy.ndarray
        Matriz inversa da massa (M⁻¹). Dimensão: (n x n).
    C_int : numpy.ndarray
        Matriz de amortecimento (C). Dimensão: (n x n).
    K_int : numpy.ndarray
        Matriz de rigidez (K). Dimensão: (n x n).
    F_int : numpy.ndarray
        Vetor de forças externas aplicadas (F(t)). Dimensão: (n,).

    Retorno:
    -------
    numpy.ndarray
        Derivadas do vetor de estado, concatenando:
            dx/dt = [velocidades (x'); acelerações (x'')].
        Dimensão: (2n,).

    Exemplo:
    --------
    Considere um sistema com 2 graus de liberdade (DoFs), onde:
        - M = [[2, 0], [0, 1]]
        - C = [[0.1, 0], [0, 0.2]]
        - K = [[3, -1], [-1, 2]]
        - F(t) = [1, 0]
    Podemos simular o sistema usando um integrador como `solve_ivp`.

    Notas:
    ------
    - Para resolver numericamente, é necessário usar um integrador como
      `scipy.integrate.solve_ivp` ou `scipy.integrate.odeint`.
    """
    # Número de graus de liberdade (n)
    n_dofs = M_int_inv.shape[0]

    # Separando deslocamentos e velocidades
    deslocamentos = x[0:n_dofs]
    velocidades = x[n_dofs:]

    # Calculando as acelerações usando a equação do movimento
    aceleracoes = np.dot(M_int_inv, F_int - np.dot(C_int, velocidades) - np.dot(K_int, deslocamentos))

    # Concatenando velocidades e acelerações
    dxdt = np.concatenate((velocidades, aceleracoes))

    return dxdt


def initialize_matrices(n, m, c, k):
    M = m * np.eye((n))  # matriz de massa
    K = np.zeros((n, n))  # matriz de rigidez
    K[0, 0] = k
    C = np.zeros((n, n))  # matriz de rigidez
    C[0, 0] = c
    for ii in range(0, n - 1):
        K[ii : ii + 2, ii : ii + 2] = K[ii : ii + 2, ii : ii + 2] + np.array([[k, -k], [-k, k]])
        C[ii : ii + 2, ii : ii + 2] = C[ii : ii + 2, ii : ii + 2] + np.array([[c, -c], [-c, c]])

    return M, K, C


def solve_differential_system_forced(t, M, C, K, F, fs, x0=None, v0=None):
    M_inv = np.linalg.inv(M)
    if x0 is None:
        x0 = np.zeros((len(M),))
    if v0 is None:
        v0 = np.zeros((len(M),))
    x_num = np.zeros((len(M), len(t)))
    v_num = np.zeros((len(M), len(t)))
    x_num[:, 0], v_num[:, 0] = x0, v0

    for i in range(1, len(t)):
        y0_rk = np.concatenate((x_num[:, i - 1], v_num[:, i - 1]), axis=0)
        F_aux = (F[:, i - 1] + F[:, i]) / 2
        aux = integrate.solve_ivp(
            fun=Linear_System_MDoF,
            t_span=t[i - 1 : i + 1],
            y0=y0_rk,
            method="RK45",
            t_eval=np.array([t[i]]),
            args=(M_inv, C, K, F_aux),
            max_step=1 / fs,
        )
        x_num[:, i] = aux.y[0 : M_inv.shape[0], 0]
        v_num[:, i] = aux.y[M_inv.shape[0] :, 0]
    return x_num, v_num


def simulate_free_response(M, C, K, F, x0, v0, t, method="RK45"):
    """
    Simula o sistema dinâmico de múltiplos graus de liberdade (MDoF).

    Parâmetros:
    ----------
    M : numpy.ndarray
        Matriz de massa (n x n).
    C : numpy.ndarray
        Matriz de amortecimento (n x n).
    K : numpy.ndarray
        Matriz de rigidez (n x n).
    F : numpy.ndarray
        Vetor de forças externas aplicadas (n,).
    x0 : numpy.ndarray
        Vetor de deslocamentos iniciais (n,).
    v0 : numpy.ndarray
        Vetor de velocidades iniciais (n,).
    t : numpy.ndarray
        Vetor de instantes de tempo para a simulação.
    method : str
        Método de integração numérica (padrão: "RK45").

    Retorno:
    -------
    x_num : numpy.ndarray
        Matriz de deslocamentos ao longo do tempo (n x len(t)).
    v_num : numpy.ndarray
        Matriz de velocidades ao longo do tempo (n x len(t)).
    """
    M_inv = np.linalg.inv(M)
    cond_iniciais = np.concatenate((x0, v0), axis=0)

    aux = integrate.solve_ivp(
        fun=Linear_System_MDoF,
        t_span=(t[0], t[-1]),
        y0=cond_iniciais,
        method=method,
        t_eval=t,
        args=(M_inv, C, K, F),
    )

    n_dofs = M.shape[0]
    x_num = aux.y[:n_dofs, :]
    v_num = aux.y[n_dofs:, :]

    return x_num, v_num


def simulate_forced_response(M, C, K, A, w, t, fs):
    """
    Simula a resposta forçada de um sistema dinâmico de múltiplos graus de liberdade (MDoF).

    Parâmetros:
    ----------
    M : numpy.ndarray
        Matriz de massa (n x n).
    C : numpy.ndarray
        Matriz de amortecimento (n x n).
    K : numpy.ndarray
        Matriz de rigidez (n x n).
    A : float
        Amplitude da força aplicada.
    w : float
        Frequência da força aplicada.
    t : numpy.ndarray
        Vetor de instantes de tempo para a simulação.
    fs : int
        Taxa de amostragem.

    Retorno:
    -------
    x_num : numpy.ndarray
        Matriz de deslocamentos ao longo do tempo (n x len(t)).
    v_num : numpy.ndarray
        Matriz de velocidades ao longo do tempo (n x len(t)).
    """
    # Criando uma matriz de forças
    pos_forca = 0
    F = np.zeros((len(M), len(t)))

    # Definição do vetor força
    F[pos_forca, :] = A * np.sin(2 * np.pi * w * t)

    # Condições iniciais
    x_num, v_num = solve_differential_system_forced(t, M, C, K, F, fs)

    return x_num, v_num


def simulate_chirp_response(M, C, K, t, fs, A=1, w0=2, w1=20, pos_forca=0):
    """
    Simula a resposta de um sistema dinâmico de múltiplos graus de liberdade (MDoF) a uma força chirp.

    Parâmetros:
    ----------
    M : numpy.ndarray
        Matriz de massa (n x n).
    C : numpy.ndarray
        Matriz de amortecimento (n x n).
    K : numpy.ndarray
        Matriz de rigidez (n x n).
    t : numpy.ndarray
        Vetor de instantes de tempo para a simulação.
    fs : int
        Taxa de amostragem.
    A : float, opcional
        Amplitude da força chirp (padrão: 1).
    w0 : float, opcional
        Frequência inicial da força chirp (padrão: 2 Hz).
    w1 : float, opcional
        Frequência final da força chirp (padrão: 20 Hz).
    pos_forca : int, opcional
        Posição onde a força é aplicada (padrão: 0).

    Retorno:
    -------
    x_num : numpy.ndarray
        Matriz de deslocamentos ao longo do tempo (n x len(t)).
    v_num : numpy.ndarray
        Matriz de velocidades ao longo do tempo (n x len(t)).
    """
    # Criando a matriz de forças
    F = np.zeros((len(M), len(t)))
    t1 = np.max(t)
    F[pos_forca, :] = A * signal.chirp(t, w0, t1, w1, method="linear", phi=0)

    # Simulando o sistema
    x_num, v_num = solve_differential_system_forced(t, M, C, K, F, fs)

    return x_num, v_num


def calculate_spectral_density(x_num, t, fs):
    """
    Calcula a densidade espectral de potência (PSD) dos deslocamentos ao longo do tempo.

    Parâmetros:
    ----------
    x_num : numpy.ndarray
        Matriz de deslocamentos ao longo do tempo (n x len(t)).
    t : numpy.ndarray
        Vetor de instantes de tempo para a simulação.
    fs : int
        Taxa de amostragem.

    Retorno:
    -------
    fx : numpy.ndarray
        Frequências associadas à PSD.
    Sxx : numpy.ndarray
        Densidade espectral de potência (PSD) dos deslocamentos.
    """
    n_window = len(t)
    Sxx = np.zeros((x_num.shape[0], n_window // 2 + 1))
    for i in range(x_num.shape[0]):
        fx, Sxx[i, :] = signal.welch(
            x_num[i, :],
            fs=fs,
            window="boxcar",
            nperseg=n_window,
            noverlap=0,
            nfft=n_window,
            detrend=False,
            return_onesided=True,
            scaling="density",
            axis=-1,
            average="mean",
        )
    return fx, Sxx


def calculate_frequency_response(M, C, K, w):
    """
    Calcula a resposta em frequência de um sistema de múltiplos graus de liberdade (MDoF).

    Parâmetros:
    ----------
    M : numpy.ndarray
        Matriz de massa (n x n).
    C : numpy.ndarray
        Matriz de amortecimento (n x n).
    K : numpy.ndarray
        Matriz de rigidez (n x n).
    w : numpy.ndarray
        Vetor de frequências em Hz.

    Retorno:
    -------
    H : numpy.ndarray
        Matriz de resposta em frequência (n x n x len(w)).
    """
    H = np.zeros((len(M), len(M), len(w)), dtype="complex")
    for n in range(0, len(w)):
        H[:, :, n] = np.linalg.inv(K - M * ((2 * np.pi * w[n]) ** 2) + complex(0, 1) * C * 2 * np.pi * w[n])
    return H


def calcular_frf_experimental(F_random, x_num, fs, janela, n_window, overl, n_fft):
    # Densidade espectral considerando correlação entrada/entrada:
    f2, P_FF = signal.csd(
        F_random,
        F_random,
        fs=fs,
        window=janela,
        nperseg=n_window,
        noverlap=overl,
        nfft=n_fft,
    )
    P_Fx = []
    P_xx = []

    for i in range(x_num.shape[0]):
        # Densidade espectral considerando correlação entrada/saída:
        f2, P_Fxi = signal.csd(
            F_random,
            x_num[i, :],
            fs=fs,
            window=janela,
            nperseg=n_window,
            noverlap=overl,
            nfft=n_fft,
        )
        # Densidade espectral considerando correlação saída/saída:
        P_Fx.append(P_Fxi)
        f2, P_xxi = signal.csd(
            x_num[i, :],
            x_num[i, :],
            fs=fs,
            window=janela,
            nperseg=n_window,
            noverlap=overl,
            nfft=n_fft,
        )
        P_xx.append(P_xxi)

    H_exp = [P_Fx[i] / P_FF for i in range(len(P_Fx))]
    coherence = [abs(P_Fx[i]) ** 2 / (P_FF * P_xx[i]) for i in range(len(P_Fx))]
    return f2, H_exp, coherence


def plot_results(t, x_num, figsize):
    fig, ax = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))
    ax[0].plot(t, x_num[0, :], "b", linewidth=2, label="RK45")
    ax[0].grid(visible=True, which="major", axis="both")
    ax[0].set_xlim([0, t[-1]])
    ax[0].set_xlabel("$t$ [s]")
    ax[0].set_ylabel("$x_1(t)$ [m]")
    ax[0].set_title("Movimento da Massa 1")
    ax[1].plot(t, x_num[1, :], "b", linewidth=2, label="RK45")
    ax[1].grid(visible=True, which="major", axis="both")
    ax[1].set_xlim([0, t[-1]])
    ax[1].set_xlabel("$t$ [s]")
    ax[1].set_ylabel("$x_2(t)$ [m]")
    ax[1].set_title("Movimento da Massa 2")
    plt.show()


def plot_frequency_response(t, x_num, fs, figsize=(6, 6)):
    """
    Plota a resposta em frequência da simulação.

    Parâmetros:
    ----------
    t : numpy.ndarray
        Vetor de instantes de tempo para a simulação.
    x_num : numpy.ndarray
        Matriz de deslocamentos ao longo do tempo (n x len(t)).
    fs : int
        Taxa de amostragem.
    figsize : tuple
        Tamanho da figura (padrão: (6, 6)).
    """
    fx, Sxx = calculate_spectral_density(x_num, t, fs)

    fig, ax = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))
    ax[0].semilogy(fx, Sxx[0, :], "b", linewidth=2)
    ax[0].grid(visible=True, which="major", axis="both")
    ax[0].set_xlim([0, 50])
    ax[0].set_xlabel("$f$ [Hz]")
    ax[0].set_ylabel("Amplitude [m$^2$/Hz]")
    ax[0].set_title("Movimento da Massa 1 (Frequência)")
    ax[1].semilogy(fx, Sxx[1, :], "b", linewidth=2)
    ax[1].grid(visible=True, which="major", axis="both")
    ax[1].set_xlim([0, 50])
    ax[1].set_xlabel("$f$ [Hz]")
    ax[1].set_ylabel("Amplitude [m$^2$/Hz]")
    ax[1].set_title("Movimento da Massa 2 (Frequência)")
    plt.show()


def plot_spectrogram(t, M, x_num, fs, figsize):
    Xxx = []
    for i in range(len(M)):
        fspec, tspec, aux = signal.spectrogram(
            x_num[i, :],
            fs=fs,
            window="boxcar",
            nperseg=len(t) // 10,
            noverlap=len(t) // 11,
            nfft=len(t),
            detrend=False,
            return_onesided=True,
            scaling="density",
            axis=-1,
            mode="psd",
        )
        Xxx.append(aux)
    Xxx = np.array(Xxx)

    fig, ax = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))
    for i in range(2):
        pc = ax[i].pcolormesh(
            tspec,
            fspec,
            Xxx[i, :],
            alpha=0.8,
            norm="log",
            cmap="jet",
            shading="auto",
        )
        fig.colorbar(pc, ax=ax[i], label="Amplitude [m$^2$/Hz]")
        ax[i].set_ylim([0, 50])
        ax[i].set_xlabel("$t$ [s]")
        ax[i].set_ylabel("$f$ [Hz]")
        ax[i].set_title("Diagrama Tempo vs Frequência")
    plt.show()


def plot_n_frf(w, H, diagonal_only=False, figsize=(6, 6), xlim=(0, 50), ylim=(1e-5, 1e-1), **plot_kwargs):
    """
    Plota a Função de Resposta em Frequência (FRF) para um sistema com n graus de liberdade.

    Parâmetros:
    ----------
    w : array-like
        Array de valores de frequência em Hz.
    H : ndarray
        Array multidimensional contendo os valores da FRF.
        H deve ter dimensões (n, n, len(w)), onde n é o número de graus de liberdade.
    diagonal_only : bool, opcional
        Se True, plota apenas a diagonal principal de H. O padrão é False.
    figsize : tuple, opcional
        Tupla especificando o tamanho de cada subplot. O padrão é (6, 6).

    Retorno:
    -------
    Nenhum: Esta função não retorna nenhum valor. Ela exibe os gráficos.

    Notas:
    ------
    - A função cria uma grade de subplots com dimensões (n, n), onde n é o número de graus de liberdade.
    - Cada subplot corresponde à magnitude da FRF entre dois graus de liberdade.
    - O eixo x representa a frequência em Hz, e o eixo y representa a magnitude da FRF em m/N.
    - Os gráficos são exibidos em escala logarítmica para o eixo y.
    - O eixo x é limitado ao intervalo [0, 50] Hz.
    - O eixo y é limitado ao intervalo [1e-5, 1e-1] m/N.
    """
    n = len(H)
    default_kwargs = {"color": "b", "linestyle": "-"}
    final_kwargs = {**default_kwargs, **plot_kwargs}
    if diagonal_only:
        fig, ax = plt.subplots(n, 1, figsize=(figsize[0], n * figsize[1]), squeeze=False)
        ax = ax.flatten()
        for i in range(n):
            ax[i].grid(True)
            ax[i].semilogy(w, abs(H[i, i, :]), **final_kwargs)
            ax[i].set(
                xlabel=r"$\omega$ [Hz]",
                ylabel=rf"$|H_{{{i + 1}{i + 1}}}(\omega)|$ [m/N]",
                xlim=xlim,
                ylim=ylim,
            )
    else:
        fig, ax = plt.subplots(n, n, figsize=(n * figsize[0], n * figsize[1]))
        for i in range(n):
            for j in range(n):
                ax[i, j].grid(True)
                ax[i, j].semilogy(w, abs(H[i, j, :]), **final_kwargs)
                ax[i, j].set(
                    xlabel=r"$\omega$ [Hz]",
                    ylabel=rf"$|H_{{{i + 1}{j + 1}}}(\omega)|$ [m/N]",
                    xlim=xlim,
                    ylim=ylim,
                )
        plt.tight_layout()
        plt.show()


def plot_n_phase(w, H, diagonal_only=False, figsize=(6, 6), xlim=(0, 50), **plot_kwargs):
    """
    Plota a fase das funções de resposta em frequência (FRF) para um sistema de n graus de liberdade.

    Parâmetros:
    -----------
    w : array-like
        Frequências em Hz.
    H : array-like
        Matriz de funções de resposta em frequência (FRF) complexas, onde H[i, j, :] representa a FRF entre o i-ésimo e j-ésimo graus de liberdade.
    figsize : tuple, opcional
        Tamanho da figura (largura, altura) em polegadas. O padrão é (14, 6).

    Retorna:
    --------
    None
        A função não retorna nada. Ela exibe um gráfico com as fases das FRFs.
    """
    n = len(H)
    default_kwargs = {"color": "b", "linestyle": "-"}
    final_kwargs = {**default_kwargs, **plot_kwargs}

    if diagonal_only:
        fig, ax = plt.subplots(n, 1, figsize=(figsize[0], n * figsize[1]), squeeze=False)
        ax = ax.flatten()
        for i in range(n):
            ax[i].grid(True)
            ax[i].plot(w, np.angle(H[i, i, :]), **final_kwargs)
            ax[i].set(
                xlabel=r"$\omega$ [Hz]",
                ylabel=rf"$H_{{{i + 1}{i + 1}}}(\omega)$ - Fase",
                xlim=xlim,
            )
    else:
        fig, ax = plt.subplots(n, n, figsize=(n * figsize[0], n * figsize[1]))
        for i in range(n):
            for j in range(n):
                ax[i, j].grid(True)
                ax[i, j].plot(w, np.angle(H[i, j, :]), **final_kwargs)
                ax[i, j].set(xlabel=r"$\omega$ [Hz]", ylabel=rf"$H_{{{i + 1}{j + 1}}}(\omega)$ - Fase", xlim=xlim)
    plt.tight_layout()
    plt.show()


def plot_n_imaginary(w, H, diagonal_only=False, figsize=(6, 6), xlim=(0, 50), **plot_kwargs):
    """
    Plota as partes imaginárias de uma matriz de funções de transferência.

    Parâmetros:
    w (array-like): Frequências em Hz.
    H (array-like): Matriz de funções de transferência complexas.
    figsize (tuple, opcional): Tamanho da figura (largura, altura). Padrão é (14, 6).
    diagonal_only (bool, opcional): Se True, plota apenas a diagonal principal de H. Padrão é False.

    Retorna:
    None
    """
    n = len(H)
    default_kwargs = {"color": "b", "linestyle": "-"}
    final_kwargs = {**default_kwargs, **plot_kwargs}  # Combina defaults com kwargs do usuário

    if diagonal_only:
        fig, ax = plt.subplots(n, 1, figsize=(figsize[0], n * figsize[1]), squeeze=False)
        ax = ax.flatten()
        for i in range(n):
            ax[i].grid(visible=True, which="major", axis="both")
            ax[i].plot(w, np.imag(H[i, i, :]), **final_kwargs)
            ax[i].set(xlabel=r"$\omega$ [Hz]", ylabel=rf"imag($H_{{{i + 1}{i + 1}}}(\omega)$)", xlim=xlim)
    else:
        fig, ax = plt.subplots(n, n, figsize=(n * figsize[0], n * figsize[1]))
        for i in range(n):
            for j in range(n):
                ax[i, j].grid(visible=True, which="major", axis="both")
                ax[i, j].plot(w, np.imag(H[i, j, :]), **final_kwargs)
                ax[i, j].set(xlabel=r"$\omega$ [Hz]", ylabel=rf"imag($H_{{{i + 1}{j + 1}}}(\omega)$)", xlim=xlim)
    plt.tight_layout()
    plt.show()


def plot_n_coherence(f2, coherence, figsize=(14, 6)):
    """
    Plota a coerência entre sinais em uma grade de subplots.

    Parâmetros:
    f2 (array-like): Frequências em Hz.
    coherence (array-like): Matriz de coerência com dimensões (n, n, m), onde n é o número de sinais e m é o número de pontos de frequência.
    figsize (tuple, opcional): Tamanho da figura (largura, altura) em polegadas. O padrão é (14, 6).

    Retorna:
    None: A função exibe o gráfico mas não retorna nenhum valor.
    """
    fig, ax = plt.subplots(len(coherence), len(coherence), figsize=figsize)
    for i in range(len(coherence)):
        for j in range(len(coherence)):
            ax[i, j].grid(visible=True, which="major", axis="both")
            ax[i, j].plot(f2, coherence[i, j, :], "b")
            ax[i, j].set(xlabel=r"$\omega$ [Hz]", ylabel="Coerência")
            ax[i, j].set_xlim(0, 50)
    plt.show()
