use image::{GenericImageView, ImageBuffer};
use ndarray::prelude::*;
use ndarray_linalg::Solve;

struct Filter {
    pub weights: Vec<FilterWeight>,
}

struct FilterWeight {
    pub dx: isize,
    pub dy: isize,
    pub weight: f32,
}

struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub sigma: f32,
    pub response: f32,
}

fn rescale_sigma(linear_sigma: f32) -> f32 {
    0.82357472 * (0.68797398 * linear_sigma).exp()
}

fn get_gaussian_kernel(ksize: isize, sigma: f32) -> Vec<f32> {
    assert!(ksize % 2 == 1);

    let hsize = (ksize - 1) / 2;

    let mut arr = (0..=ksize)
        .map(|i| {
            let n = (i - hsize).pow(2) as f32;
            (-n / (2.0 * sigma.powi(2))).exp()
        })
        .collect::<Vec<f32>>();

    let sum = arr.iter().fold(0.0, |sum, a| sum + a);
    for item in &mut arr {
        *item /= sum;
    }
    arr
}

fn main() {
    let scale_depth: isize = 3;
    let extrema_threshold: f32 = 0.05;
    let lower_cm_threshold: f32 = 0.7;
    let upper_cm_threshold: f32 = 1.5;

    let kernel = get_gaussian_kernel(5, 0.6);
    let h_filters = {
        let mut filters = vec![];

        let mut weights = vec![];
        for y in -2isize..=2 {
            for x in -2isize..=2 {
                weights.push(FilterWeight {
                    dx: x,
                    dy: y,
                    weight: kernel[(2 + x) as usize] * kernel[(2 + y) as usize],
                });
            }
        }
        filters.push(Filter { weights });

        let base_spline = [1. / 16., 4. / 16., 6. / 16., 4. / 16., 1. / 16.];
        for i in 0isize..scale_depth + 2 {
            let mut weights = vec![];

            let pitch = (2i32.pow(i as u32) - 1) as isize;
            for y in -2isize..=2 {
                for x in -2isize..=2 {
                    weights.push(FilterWeight {
                        dx: (pitch + 1) * x,
                        dy: (pitch + 1) * y,
                        weight: base_spline[(2 + x) as usize] * base_spline[(2 + y) as usize],
                    });
                }
            }
            filters.push(Filter { weights });
        }
        filters
    };

    // load image
    let dynamic_img = image::open("./graf1.png").unwrap();
    let dim = dynamic_img.dimensions();
    println!("Image size: {:?}", dim);

    let img = dynamic_img.grayscale();
    let img = img.as_luma8().unwrap();

    let mut imgbuf = ImageBuffer::new(dim.0, dim.1);
    for (x, y, pixel) in img.enumerate_pixels() {
        let val = pixel[0] as f32 / 255.;
        let target_pixel = imgbuf.get_pixel_mut(x, y);
        *target_pixel = image::Luma([val]);
    }

    let coarse_imgs = (0..=scale_depth + 2).fold(vec![imgbuf; 1], |mut data, d| {
        let mut new_img = ImageBuffer::new(dim.0, dim.1);
        for (x, y, target_pixel) in new_img.enumerate_pixels_mut() {
            let mut val = 0 as f32;
            for info in &h_filters[d as usize].weights {
                let tx = x as isize + info.dx;
                let ty = y as isize + info.dy;

                let w = info.weight;
                if 0 <= ty && ty < dim.1 as isize && 0 <= tx && tx < dim.0 as isize {
                    let image::Luma(prev_val) = &data[d as usize].get_pixel(tx as u32, ty as u32);
                    val += w * prev_val[0];
                }
            }
            *target_pixel = image::Luma([val]);
        }
        data.push(new_img);
        data
    });

    //
    for (i, img) in coarse_imgs.iter().enumerate() {
        let mut imgbuf = ImageBuffer::new(dim.0, dim.1);
        for (x, y, pixel) in img.enumerate_pixels() {
            let val = (pixel[0] * 255.) as u8;

            let target_pixel = imgbuf.get_pixel_mut(x, y);
            *target_pixel = image::Luma([val]);
        }
        imgbuf.save(format!("coarse_img_{}.png", i)).unwrap();
    }

    let fine_imgs = {
        let mut imgs = vec![];
        for i in 0..coarse_imgs.len() - 1 {
            let mut imgbuf = ImageBuffer::new(dim.0, dim.1);
            for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
                *pixel = image::Luma([
                    coarse_imgs[i].get_pixel(x, y)[0] - coarse_imgs[i + 1].get_pixel(x, y)[0]
                ]);
            }
            imgs.push(imgbuf);
        }
        imgs
    };

    //
    for (i, img) in fine_imgs.iter().enumerate() {
        let mut imgbuf = ImageBuffer::new(dim.0, dim.1);
        for (x, y, pixel) in img.enumerate_pixels() {
            let val = (pixel[0] * 255.).abs() as u8;

            let target_pixel = imgbuf.get_pixel_mut(x, y);
            *target_pixel = image::Luma([val]);
        }
        imgbuf.save(format!("fine_img_{}.png", i)).unwrap();
    }

    let mut keypoints: Vec<KeyPoint> = vec![];
    for d in 0..(fine_imgs.len() - 2) {
        let pre_img = &fine_imgs[d];
        let cur_img = &fine_imgs[d + 1];
        let nxt_img = &fine_imgs[d + 2];
        let margin = 2u32.pow((d + 1) as u32);
        for (x, y, pixel) in cur_img.enumerate_pixels() {
            if !(margin <= x && x < dim.0 - margin && margin <= y && y < dim.1 - margin) {
                continue;
            }
            let cur11 = pixel[0];
            if cur11 < extrema_threshold - 0.01 {
                continue;
            }
            let cur00 = cur_img.get_pixel(x - 1, y - 1)[0];
            let cur01 = cur_img.get_pixel(x - 1, y)[0];
            let cur02 = cur_img.get_pixel(x - 1, y + 1)[0];
            let cur10 = cur_img.get_pixel(x, y - 1)[0];
            let cur12 = cur_img.get_pixel(x, y + 1)[0];
            let cur20 = cur_img.get_pixel(x + 1, y - 1)[0];
            let cur21 = cur_img.get_pixel(x + 1, y)[0];
            let cur22 = cur_img.get_pixel(x + 1, y + 1)[0];

            let nxt01 = nxt_img.get_pixel(x - 1, y)[0];
            let nxt11 = nxt_img.get_pixel(x, y)[0];
            let nxt21 = nxt_img.get_pixel(x + 1, y)[0];
            let nxt10 = nxt_img.get_pixel(x, y - 1)[0];
            let nxt12 = nxt_img.get_pixel(x, y + 1)[0];

            let pre01 = pre_img.get_pixel(x - 1, y)[0];
            let pre11 = pre_img.get_pixel(x, y)[0];
            let pre21 = pre_img.get_pixel(x + 1, y)[0];
            let pre10 = pre_img.get_pixel(x, y - 1)[0];
            let pre12 = pre_img.get_pixel(x, y + 1)[0];
            let vec_dd = arr1(&[
                (cur21 - cur01) / 2.,
                (cur12 - cur10) / 2.,
                (nxt11 - pre11) / 2.,
            ]);
            let h00 = cur21 + cur01 - 2. * cur11;
            let h11 = cur12 + cur10 - 2. * cur11;
            let h22 = nxt11 + pre11 - 2. * cur11;
            let h01 = (cur22 - cur02 - cur20 + cur00) / 4.;
            let h02 = (nxt21 - pre21 - nxt01 + pre01) / 4.;
            let h12 = (nxt12 - pre12 - nxt10 + pre10) / 4.;
            let mat_h = arr2(&[[h00, h01, h02], [h01, h11, h12], [h02, h12, h22]]);

            let dpos = mat_h.solve(&vec_dd).unwrap();

            if dpos[0].abs() < 0.5 && dpos[1].abs() < 0.5 && dpos[2].abs() < 0.5 {
                let response = cur11 + vec_dd.dot(&dpos) / 2.;
                let cm = 1. - 4. * (h00 * h11 - h01.powi(2)) / (h00 + h11).powi(2);
                if response > extrema_threshold
                    && (cm <= lower_cm_threshold || upper_cm_threshold <= cm)
                {
                    keypoints.push(KeyPoint {
                        x: x as f32 + dpos[0],
                        y: y as f32 + dpos[1],
                        sigma: rescale_sigma((d + 1) as f32 + dpos[2]),
                        response,
                    });
                }
            }
        }
    }
    println!("{} keypoints were found", keypoints.len());

    let mut imgbuf = ImageBuffer::new(dim.0, dim.1);
    for (x, y, pixel) in img.enumerate_pixels() {
        let val = pixel[0];
        let target_pixel = imgbuf.get_pixel_mut(x, y);
        *target_pixel = image::Rgb([val, val, val]);
    }
    for kp in keypoints {
        // println!("({}, {}) {} {}", kp.x, kp.y, kp.sigma, kp.response);
        let target_pixel = imgbuf.get_pixel_mut(kp.x.round() as u32, kp.y.round() as u32);
        *target_pixel = image::Rgb([255, 0, 0]);
    }
    imgbuf.save("result.png").unwrap();
}
