use nannou::prelude::*;
use nannou::{image};
use ndarray::{s, Array2, ArrayView2};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const PI: f64 = 3.1415926535;
const LENGTH: f64 = 2.0;
const SPEED: f64 = 2.0;
const ANGLE_SPEED: f64 = PI * 2.0 * 0.1;
const DECAY: f64 = 0.02;

fn main() {
    nannou::app(model).update(update).run();
}

struct Organ {
    x: f64,
    y: f64,
    angle: f64,
}

struct Model {
    agents: Vec<Organ>,
    pheromone: Array2<f64>,
    texture: wgpu::Texture,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(WIDTH, HEIGHT)
        .view(view)
        .key_pressed(key_pressed)
        .key_released(key_released)
        .build()
        .unwrap();

    let window = app.main_window();
    let win = window.rect();
    let texture = wgpu::TextureBuilder::new()
        .size([win.w() as u32, win.h() as u32])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING)
        .build(window.device());

    let pheromone = Array2::from_elem((WIDTH as usize, HEIGHT as usize), 0.0);
    let mut agents = vec![];
    for _ in 0..10000 {
        let x = random::<f64>() * WIDTH as f64;
        let y = random::<f64>() * HEIGHT as f64;
        let angle = random::<f64>() * 2.0 * PI;
        agents.push(Organ { x, y, angle })
    }

    Model {
        agents,
        pheromone,
        texture,
    }
}

fn loop_coord(a: f64, min: f64, max: f64) -> f64 {
    if a > max {
        return loop_coord(a - (max - min), min, max);
    }
    if a < min {
        return loop_coord(a + (max - min), min, max);
    }
    return a;
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    // roll
    for a in &mut model.agents {
        let forward = model.pheromone.get((
            (a.x + a.angle.cos() * LENGTH) as usize,
            (a.y + a.angle.sin() * LENGTH) as usize,
        ));
        let left = model.pheromone.get((
            (a.x + (a.angle - ANGLE_SPEED).cos() * LENGTH) as usize,
            (a.y + (a.angle - ANGLE_SPEED).sin() * LENGTH) as usize,
        ));
        let right = model.pheromone.get((
            (a.x + (a.angle + ANGLE_SPEED).cos() * LENGTH) as usize,
            (a.y + (a.angle + ANGLE_SPEED).sin() * LENGTH) as usize,
        ));
        if let Some(f) = forward {
            if let Some(r) = right {
                if let Some(l) = left {
                    if f < l && f > r {
                        a.angle -= PI * 2.0 * 0.05;
                    } else if f > l && f < r {
                        a.angle += PI * 2.0 * 0.05;
                    }
                }
            }
        }
    }

    // move
    for a in &mut model.agents {
        let dx = a.angle.cos() * SPEED;
        let dy = a.angle.sin() * SPEED;
        a.x += dx;
        a.y += dy;
        a.x = loop_coord(a.x, 0.0, 512.0);
        a.y = loop_coord(a.y, 0.0, 512.0);
    }

    // pheromone
    for a in &model.agents {
        let pix = model
            .pheromone
            .get_mut((a.x as usize, a.y as usize))
            .unwrap();
        *pix += 0.3
    }
    // diffsion
    let r = lapla(&model.pheromone.view(), 2.0);
    model.pheromone += &r.mapv(|a: f64| a * 0.1);
    // decay
    model.pheromone = model.pheromone.mapv(|a: f64| a * (1.0 - DECAY));
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);

    let win = app.window_rect();

    let pixel: Vec<u8> = model
        .pheromone
        .iter()
        .map(|i| {
            [
                (i * 128.0) as u8,
                (i * 128.0) as u8,
                (i * 128.0) as u8,
                std::u8::MAX,
            ]
        })
        .flatten()
        .collect();
    let image: Option<image::ImageBuffer<image::Rgba<u8>, Vec<_>>> =
        image::ImageBuffer::from_vec(win.w() as u32, win.h() as u32, pixel);

    if let Some(image_unwrap) = image {
        let flat_samples = image_unwrap.as_flat_samples();
        model.texture.upload_data(
            app.main_window().device(),
            &mut *frame.command_encoder(),
            &flat_samples.as_slice(),
        );

        let draw = app.draw();
        draw.texture(&model.texture);

        // Write to the window frame.
        draw.to_frame(app, &frame).unwrap();
    }
}

fn key_released(app: &App, model: &mut Model, key: Key) {
    match key {
        Key::S => {
            app.main_window()
                .capture_frame(app.exe_name().unwrap() + ".png");
        }
        Key::R => {
            model.pheromone = Array2::from_elem((WIDTH as usize, HEIGHT as usize), 0.0);
            let mut agents = vec![];
            for _ in 0..10000 {
                let x = random::<f64>() * WIDTH as f64;
                let y = random::<f64>() * HEIGHT as f64;
                let angle = random::<f64>() * 2.0 * PI;
                agents.push(Organ { x, y, angle })
            }
            model.agents = agents;
        }
        _otherkey => (),
    }
}

fn key_pressed(_app: &App, _model: &mut Model, key: Key) {
    match key {
        Key::Up => {
            // model.falloff += 0.05;
        }
        Key::Down => {
            // model.falloff -= 0.05;
        }
        Key::Left => {
            // model.octaves -= 1;
        }
        Key::Right => {
            // model.octaves += 1;
        }
        _otherkey => (),
    }
}

fn roll<A>(a: &ArrayView2<A>, dir: &[isize; 2], elem: A) -> Array2<A>
where
    A: Clone,
{
    let mut b = Array2::from_elem(a.dim(), elem);
    let x = dir[0];
    let y = dir[1];
    if x == 0 {
        if y == 0 {
            b.assign(&a);
        } else {
            b.slice_mut(s![.., y..]).assign(&a.slice(s![.., ..-y]));
            b.slice_mut(s![.., ..y]).assign(&a.slice(s![.., -y..]));
        }
    } else {
        if y == 0 {
            b.slice_mut(s![x.., ..]).assign(&a.slice(s![..-x, ..]));
            b.slice_mut(s![..x, ..]).assign(&a.slice(s![-x.., ..]));
        } else {
            b.slice_mut(s![x.., y..]).assign(&a.slice(s![..-x, ..-y]));
            b.slice_mut(s![..x, y..]).assign(&a.slice(s![-x.., ..-y]));
            b.slice_mut(s![x.., ..y]).assign(&a.slice(s![..-x, -y..]));
            b.slice_mut(s![..x, ..y]).assign(&a.slice(s![-x.., -y..]));
        }
    }
    b
}

fn lapla(x: &ArrayView2<f64>, dx: f64) -> Array2<f64> {
    let mut r = Array2::zeros(x.dim());
    let ux_r = roll(x, &[1, 0], 0.0);
    let ux_l = roll(x, &[-1, 0], 0.0);
    let uy_r = roll(x, &[0, 1], 0.0);
    let uy_l = roll(x, &[0, -1], 0.0);
    ndarray::Zip::from(&mut r)
        .and(x)
        .and(&ux_r)
        .and(&uy_r)
        .and(&ux_l)
        .and(&uy_l)
        .for_each(|r, &x, &xx, &xy, &xxl, &xyl| {
            *r = (xx + xy + xxl + xyl - 4.0 * x) / dx / dx;
        });
    r
}
