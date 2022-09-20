use anyhow::Result;
use cfilt::{deposit_particles, Currents, Particles};
use clap::Parser;
use hdf5::File;
use ndarray::{s, Array1, Array2, Ix3};

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
    /// Particle velocity
    #[clap(long, default_value = "0.5")]
    c: f32,
    /// Compress output file
    #[clap(default_value = "false")]
    compress: bool,
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
    let j_shape = _jx.slice(s![0, .., ..]).raw_dim();

    let jx = Array2::zeros(j_shape);
    let jy = Array2::zeros(j_shape);
    let jz = Array2::zeros(j_shape);

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
    let mut c = Currents { jx, jy, jz };

    deposit_particles(&mut p, &mut c, mx, my, args.mghost, args.c);

    param_file.close()?;
    field_file.close()?;
    particle_file.close()?;

    let output_file = File::create(args.output_file)?;

    macro_rules! write_dataset {
        ($name:ident, $compress:expr) => {
            let builder = output_file.new_dataset_builder().with_data(&c.$name);
            if $compress {
                builder.blosc_zstd(9, true).create(stringify!($name))?;
            } else {
                builder.create(stringify!($name))?;
            }
        };
    }

    write_dataset!(jx, args.compress);
    write_dataset!(jy, args.compress);
    write_dataset!(jz, args.compress);

    output_file.close()?;

    Ok(())
}
