"""Go2 proprioceptive velocity math, shared verbatim by training and deployment.

q,dq: [...,4,3] FR/FL/RR/RL, omega: [...,3], contacts: [...,4].
Backend is numpy or torch. Constants match go2_x5.urdf, units m/rad/s.
"""
import math


def _stack(items, xp):
    return xp.stack(items, dim=-1) if xp.__name__ == 'torch' else xp.stack(items, axis=-1)


def _constant(q, values, xp):
    return q.new_tensor(values) if xp.__name__ == 'torch' else xp.asarray(values, dtype=q.dtype)


def foot_kinematics(q, dq, xp):
    a,t,k=q[...,0],q[...,1],q[...,2]
    da,dt,dk=dq[...,0],dq[...,1],dq[...,2]
    side=_constant(q,[-1,1,-1,1],xp)
    front=_constant(q,[1,1,-1,-1],xp)
    h=side*.0955; length=.213
    x=-length*(xp.sin(t)+xp.sin(t+k))
    z0=-length*(xp.cos(t)+xp.cos(t+k))
    dx=-length*(xp.cos(t)*dt+xp.cos(t+k)*(dt+dk))
    dz0=length*(xp.sin(t)*dt+xp.sin(t+k)*(dt+dk))
    y=h*xp.cos(a)-z0*xp.sin(a)
    z=h*xp.sin(a)+z0*xp.cos(a)
    dy=-h*xp.sin(a)*da-dz0*xp.sin(a)-z0*xp.cos(a)*da
    dz=h*xp.cos(a)*da+dz0*xp.cos(a)-z0*xp.sin(a)*da
    return _stack([x+front*.1934,y+side*.0465,z],xp),_stack([dx,dy,dz],xp)


def contact_velocity(q, dq, omega, contacts, previous, dt, xp):
    """-J*dq-omega×r, reject inconsistent support legs, then 60 ms low-pass.

    No contact: hold previous estimate (caller must gate prolonged loss).
    Agreement cannot detect common-mode slip of all support feet.
    """
    r,dr=foot_kinematics(q,dq,xp)
    w=omega[...,None,:]
    cross=_stack([w[...,1]*r[...,2]-w[...,2]*r[...,1],
                  w[...,2]*r[...,0]-w[...,0]*r[...,2],
                  w[...,0]*r[...,1]-w[...,1]*r[...,0]],xp)
    estimates=-dr-cross
    delta=estimates[..., :,None,:]-estimates[...,None,:,:]
    agree=xp.sum(delta*delta,axis=-1) <= .2**2
    votes=xp.sum(agree & contacts[...,None,:],axis=-1)
    scores=xp.where(contacts,votes,-1)
    best=xp.argmax(scores,axis=-1)
    selector=_constant(q,[0,1,2,3],xp)==best[...,None]
    reference=xp.sum(estimates*selector[...,None],axis=-2)
    error=estimates-reference[...,None,:]
    accepted=contacts & (xp.sum(error*error,axis=-1)<=.2**2)
    count=xp.sum(accepted,axis=-1)
    denom=xp.maximum(count,_constant(q,1,xp))
    measured=xp.sum(estimates*accepted[...,None],axis=-2)/denom[...,None]
    velocity=previous+(1-math.exp(-dt/.06))*(measured-previous)
    return xp.where((count>0)[...,None],velocity,previous),count
