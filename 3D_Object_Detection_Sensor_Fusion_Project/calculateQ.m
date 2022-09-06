function Q = calculateQ()
syms dt q

F = eye(6);
F(1,4) = 7;
F(2,5) = 7;
F(3,6) = 7;

Q = zeros(6);
Q(4,4) = 3;
Q(5,5) = 3;
Q(6,6) = 3;

Q = F*Q*F'

end