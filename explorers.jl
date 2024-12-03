# No adaptation
# info = (dimension, step size)
struct RWMH{T}
    dimension::Int
    step::T
end

RWMH(d::Int) = RWMH(d,1.0)


# AM adaptation
# info = (dimension, mu, Sigma)
mutable struct AM{R,T}
    dimension::Int
    mu::R
    Sigma::T
end

AM(d::Int) = AM(d,fill(0.0,d),Diagonal(fill(1.0,d))+zeros(d,d))

# RRWMH adaptation
# info = (dimension, mu, Sigma, q vector, window length)
mutable struct RRWMH{R,T,U}
    dimension::Int
    mu::R
    Sigma::T
    q::U
    win_len::Int
end

RRWMH(d::Int,win_len::Int) = RRWMH(d,fill(0.0,d),Diagonal(fill(1.0,d))+zeros(d,d),fill(Gamma(1,1),d),win_len)