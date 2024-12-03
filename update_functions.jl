# No adaptation
function no_update!(explorer::RWMH, chain, t) end


# AM adaptation
function AM_update!(explorer::AM, chain, t)
    state = chain[t,1:explorer.dimension]
    c_state = state .- explorer.mu
    explorer.mu .= explorer.mu .+ t^(-1) .* c_state
    explorer.Sigma .= explorer.Sigma .+ t^(-1) .* (c_state*c_state' .- explorer.Sigma)
end


# RRWMH adaptation
# other_stats = (θ_(t-win_len), Σ θ, Σ θ², )
function RRWMH_update!(explorer::RRWMH, chain, t)
    state =  @view chain[t,1:explorer.dimension]
    last_state = @view chain[max(t-explorer.win_len,1),1:explorer.dimension]
    
    # update AM part of explorer
    c_state = state .- explorer.mu
    explorer.mu .= explorer.mu .+ t^(-1) .* c_state
    explorer.Sigma = explorer.Sigma + t^(-1) .* (c_state*c_state' - explorer.Sigma)
    
    # update moving averages & compute m, V
    replica.other_stats.m_vec = sqrt.(diag(explorer.Sigma)) # means of q
    if t > explorer.win_len
        replica.other_stats.moving_mean += (state - last_state) ./ explorer.win_len
        replica.other_stats.moving_mean2 += (state.^2 - last_state .^ 2) ./ explorer.win_len
        new_moving_vars = replica.other_stats.moving_mean2 - replica.other_stats.moving_mean .^ 2    
        replica.other_stats.moving_vars_mean .*= (t-explorer.win_len)/(t-explorer.win_len+1)
        replica.other_stats.moving_vars_mean2 .*= (t-explorer.win_len)/(t-explorer.win_len+1)
        replica.other_stats.moving_vars_mean += new_moving_vars ./ (t-explorer.win_len+1)
        replica.other_stats.moving_vars_mean2 += new_moving_vars .^ 2 ./ (t-explorer.win_len+1)
        if t > 100 # wait for variance estimator to stabilize
            replica.other_stats.V_vec = replica.other_stats.moving_vars_mean2 - replica.other_stats.moving_vars_mean .^ 2 
            # @show replica.other_stats.m_vec
            # @show replica.other_stats.V_vec
        end
    else
        replica.other_stats.moving_mean = vec(mean(chain[1:t,1:explorer.dimension], dims=1))
        replica.other_stats.moving_mean2 = vec(mean(chain[1:t,1:explorer.dimension].^2, dims=1))
        #replica.other_stats.V_vec = fill(1.0, explorer.dimension)
    end

    # update qj
    shape_vec = replica.other_stats.m_vec .^2 ./ replica.other_stats.V_vec
    scale_vec = replica.other_stats.V_vec ./ replica.other_stats.m_vec
    for j in 1:explorer.dimension
        explorer.q[j] = Gamma(shape_vec[j],scale_vec[j])
    end
end