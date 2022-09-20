use anyhow::Result;
use cfilt::{deposit_particles, Currents, Particles};
use clap::Parser;
use hdf5::File;
use ndarray::{s, Array1, Ix3};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Parameter file
    #[clap(long)]
    param_file: String,
    /// Field file
    #[clap(long)]
    field_file: String,
    /// Particle file
    #[clap(long)]
    particle_file: String,
    /// Output file
    #[clap(long)]
    output_file: String,
    /// Thinning factor
    #[clap(long, default_value = "1")]
    thin: usize,
    /// Number of ghost zones
    #[clap(long, default_value = "3")]
    mghost: i32,
    /// Charge value
    #[clap(long, default_value = "1.0")]
    q: f32,
    /// Particle velocity
    #[clap(long, default_value = "0.5")]
    c: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let param_file = File::open(args.param_file)?;

    let _mx: Array1<f32> = param_file.dataset("mx")?.read_1d()?;
    let mx = _mx[0] as i32;
    let _my: Array1<f32> = param_file.dataset("my")?.read_1d()?;
    let my = _my[0] as i32;

    let field_file = File::open(args.field_file)?;

    let _jx = field_file.dataset("jx")?.read::<f32, Ix3>()?;
    let jx = _jx.slice(s![0, .., ..]).to_owned();
    let _jy = field_file.dataset("jy")?.read::<f32, Ix3>()?;
    let jy = _jy.slice(s![0, .., ..]).to_owned();
    let _jz = field_file.dataset("jz")?.read::<f32, Ix3>()?;
    let jz = _jz.slice(s![0, .., ..]).to_owned();

    let particle_file = File::open(args.particle_file)?;

    let xe: Array1<f32> = particle_file.dataset("xe")?.read_1d()?;
    let ye: Array1<f32> = particle_file.dataset("ye")?.read_1d()?;
    let ze: Array1<f32> = particle_file.dataset("ze")?.read_1d()?;
    let ue: Array1<f32> = particle_file.dataset("ue")?.read_1d()?;
    let ve: Array1<f32> = particle_file.dataset("ve")?.read_1d()?;
    let we: Array1<f32> = particle_file.dataset("we")?.read_1d()?;

    let xi: Array1<f32> = particle_file.dataset("xi")?.read_1d()?;
    let yi: Array1<f32> = particle_file.dataset("yi")?.read_1d()?;
    let zi: Array1<f32> = particle_file.dataset("zi")?.read_1d()?;
    let ui: Array1<f32> = particle_file.dataset("ui")?.read_1d()?;
    let vi: Array1<f32> = particle_file.dataset("vi")?.read_1d()?;
    let wi: Array1<f32> = particle_file.dataset("wi")?.read_1d()?;

    let electrons = Particles {
        x: xe.slice(s![..;args.thin]).to_owned(),
        y: ye.slice(s![..;args.thin]).to_owned(),
        z: ze.slice(s![..;args.thin]).to_owned(),
        u: ue.slice(s![..;args.thin]).to_owned(),
        v: ve.slice(s![..;args.thin]).to_owned(),
        w: we.slice(s![..;args.thin]).to_owned(),
    };

    let ions = Particles {
        x: xi.slice(s![..;args.thin]).to_owned(),
        y: yi.slice(s![..;args.thin]).to_owned(),
        z: zi.slice(s![..;args.thin]).to_owned(),
        u: ui.slice(s![..;args.thin]).to_owned(),
        v: vi.slice(s![..;args.thin]).to_owned(),
        w: wi.slice(s![..;args.thin]).to_owned(),
    };

    let mut p = vec![electrons, ions];
    let mut c = Currents {
        jx: jx.to_owned(),
        jy: jy.to_owned(),
        jz: jz.to_owned(),
    };

    deposit_particles(&mut p, &mut c, mx, my, args.mghost, args.q, args.c);

    param_file.close()?;
    field_file.close()?;
    particle_file.close()?;

    let output_file = File::create(args.output_file)?;
    output_file
        .new_dataset_builder()
        .with_data(&c.jx)
        .create("jx")?;
    output_file
        .new_dataset_builder()
        .with_data(&c.jy)
        .create("jy")?;
    output_file
        .new_dataset_builder()
        .with_data(&c.jz)
        .create("jz")?;
    output_file.close()?;

    Ok(())
}
