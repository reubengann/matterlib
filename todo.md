1. `'Eq(W, 64*2**(1/3)*atmosphere*meter**5*(-0.259108664993958/meter**2 + 3*2**(2/3)/(8*meter**2)))'`
This equation, when called with .convert_to(joule), returns a nonsense dimension of J/meter^2. Seems
like a bug. If we evalf() it first, then convert_to(joule), we get joule.
