use ndarray::{Array1, Array2};
use std::cmp;

#[derive(Debug, Clone)]
pub struct Particles {
    pub x: Array1<f32>,
    pub y: Array1<f32>,
    pub z: Array1<f32>,
    pub u: Array1<f32>,
    pub v: Array1<f32>,
    pub w: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct Currents {
    pub jx: Array2<f32>,
    pub jy: Array2<f32>,
    pub jz: Array2<f32>,
}

pub fn deposit_particles(
    p: &mut Vec<Particles>,
    cur: &mut Currents,
    mx: i32,
    my: i32,
    mghost: i32,
    c: f32,
) {
    let maxx: f32 = (mx - mghost) as f32 + 1.;
    let minx: f32 = (mghost) as f32;
    let maxy: f32 = (my - mghost) as f32 + 1.;
    let miny: f32 = (mghost) as f32;
    let minz: f32 = (mghost) as f32;
    let maxz: f32 = (mghost) as f32 + 1.;
    let midx: f32 = 0.5 * (maxx - minx) as f32;
    let midy: f32 = 0.5 * (maxy - miny) as f32;
    let midz: f32 = 0.5 * (maxz - minz) as f32;

    for sp in 0..2 {
        let q = if sp == 0 { 1. } else { -1. };

        for n in 0..p[sp].x.len() {
            let invgam =
                1. / (1. + p[sp].u[n].powi(2) + p[sp].v[n].powi(2) + p[sp].w[n].powi(2)).sqrt();
            let x0 = p[sp].x[n] - p[sp].u[n] * invgam * c;
            let y0 = p[sp].y[n] - p[sp].v[n] * invgam * c;
            let z0 = p[sp].z[n] - p[sp].w[n] * invgam * c;

            let i1 = x0.floor() as usize;
            let i2 = p[sp].x[n].floor() as usize;
            let j1 = y0.floor() as usize;
            let j2 = p[sp].y[n].floor() as usize;
            let k1 = z0.floor() as usize;
            let k2 = p[sp].z[n].floor() as usize;

            let x1sp = x0;
            let x2sp = p[sp].x[n];
            let y1sp = y0;
            let y2sp = p[sp].y[n];
            let z1 = z0;
            let z2 = p[sp].z[n];

            let xr = f32::min(
                cmp::min(i1, i2) as f32 + 1.,
                f32::max(cmp::max(i1, i2) as f32, 0.5 * (x1sp + x2sp)),
            );
            let yr = f32::min(
                cmp::min(j1, j2) as f32 + 1.,
                f32::max(cmp::max(j1, j2) as f32, 0.5 * (y1sp + y2sp)),
            );
            let zr = f32::min(
                cmp::min(k1, k2) as f32 + 1.,
                f32::max(cmp::max(k1, k2) as f32, 0.5 * (z1 + z2)),
            );

            let k1 = 1;
            let k2 = 1;

            let Wx1 = 0.5 * (x1sp + xr) - i1 as f32;
            let Wy1 = 0.5 * (y1sp + yr) - j1 as f32;
            let Wx2 = 0.5 * (x2sp + xr) - i2 as f32;
            let Wy2 = 0.5 * (y2sp + yr) - j2 as f32;
            let Wz1 = 0.;
            let Wz2 = 0.;

            let Fx1 = -q * (xr - x1sp);
            let Fy1 = -q * (yr - y1sp);
            let Fz1 = -q * (zr - z1);
            let Fx2 = -q * (x2sp - xr);
            let Fy2 = -q * (y2sp - yr);
            let Fz2 = -q * (z2 - zr);

            let onemWx1 = 1.0 - Wx1;
            let onemWx2 = 1.0 - Wx2;
            let onemWy1 = 1.0 - Wy1;
            let onemWy2 = 1.0 - Wy2;
            let onemWz1 = 1.0 - Wz1;
            let onemWz2 = 1.0 - Wz2;

            let i1p1 = i1 as usize + 1;
            let i2p1 = i2 as usize + 1;
            let j1p1 = j1 as usize + 1;
            let j2p1 = j2 as usize + 1;

            cur.jx[[j1, i1]] += Fx1 * onemWy1 * onemWz1;
            cur.jx[[j1p1, i1]] += Fx1 * Wy1 * onemWz1;
            cur.jx[[j2, i2]] += Fx2 * onemWy2 * onemWz2;
            cur.jx[[j2p1, i2]] += Fx2 * Wy2 * onemWz2;
            cur.jy[[j1, i1]] += Fy1 * onemWx1 * onemWz1;
            cur.jy[[j1, i1p1]] += Fy1 * Wx1 * onemWz1;
            cur.jy[[j2, i2]] += Fy2 * onemWx2 * onemWz2;
            cur.jy[[j2, i2p1]] += Fy2 * Wx2 * onemWz2;
            cur.jz[[j1, i1]] += Fz1 * onemWx1 * onemWy1;
            cur.jz[[j1, i1p1]] += Fz1 * Wx1 * onemWy1;
            cur.jz[[j1p1, i1]] += Fz1 * onemWx1 * Wy1;
            cur.jz[[j1p1, i1p1]] += Fz1 * Wx1 * Wy1;
            cur.jz[[j2, i2]] += Fz2 * onemWx2 * onemWy2;
            cur.jz[[j2, i2p1]] += Fz2 * Wx2 * onemWy2;
            cur.jz[[j2p1, i2]] += Fz2 * onemWx2 * Wy2;
            cur.jz[[j2p1, i2p1]] += Fz2 * Wx2 * Wy2;

            // only periodic BCs for now;
            let perx = ((p[sp].x[n] - minx).signum() + (p[sp].x[n] - maxx).signum()) * midx;
            let pery = ((p[sp].y[n] - miny).signum() + (p[sp].y[n] - maxy).signum()) * midy;
            let perz = ((p[sp].z[n] - minz).signum() + (p[sp].z[n] - maxz).signum()) * midz;

            p[sp].x[n] -= perx;
            p[sp].y[n] -= pery;
            p[sp].z[n] -= perz;
        }
    }
}
