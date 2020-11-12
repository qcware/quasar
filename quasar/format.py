def format_statevector(
    statevector,
    real=False,
    formatstr='%+9.6f',
    cutoff=1.0E-6,
    ):

    """ Format a statevector for pretty printing, returning the resulting str.

    Params:
        statevector (np.ndarray of shape (2**N,)) - the statevector to format
        real (bool) - print only the real part of the amplitudes (False) or the
            full complex amplitudes (True - default).
        formatstr (str) - the format str for the real or imaginary part of each
            amplitude.
        cutoff (float) - amplitudes with abs less than cutoff will not be
            printed (set to 0.0 to print all amplitudes).
    Returns:
        (str) - a multiline string with the amplitudes and ket strings
            formatted nicely
    """

    N = (statevector.shape[0]&-statevector.shape[0]).bit_length()-1

    if real:
        formatstr2 = '%s' % (formatstr)
    else:
        formatstr2 = '+(%s%sj)' % (formatstr, formatstr)
    formatstr2 += '|%s>'

    lines = []
    for k, v in enumerate(statevector):
        # Cutoff
        if abs(v) < cutoff: continue
        # Ket str
        start = bin(k)[2:]
        padded = ('0' * (N - len(start))) + start
        # Strval
        if real:
            strval = formatstr2 % (v.real, padded)
        else:
            strval = formatstr2 % (v.real, v.imag, padded)
        lines.append(strval)

    return '\n'.join(lines)
        
