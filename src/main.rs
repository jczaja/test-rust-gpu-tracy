use ocl::ProQue;
use tracy_client::{frame_mark, span, Client};

fn get_start_time(event : &ocl::Event) -> ocl::Result<i64> {
    let maybe_start_time = event.profiling_info(ocl::enums::ProfilingInfo::Start)?;
    if let ocl::enums::ProfilingInfoResult::Start(start_time) = maybe_start_time{
        Ok(start_time as i64)
    } else {
        panic!("Warning: Profiling start time is zero. Ensure the command queue was created with profiling enabled.");

    }
}

fn get_end_time(event : &ocl::Event) -> ocl::Result<i64> {
    let maybe_end_time = event.profiling_info(ocl::enums::ProfilingInfo::End)?;
    if let ocl::enums::ProfilingInfoResult::End(end_time) = maybe_end_time{
        Ok(end_time as i64)
    } else {
        panic!("Warning: Profiling end time is zero. Ensure the command queue was created with profiling enabled.");

    }
}


fn main() -> ocl::Result<()> {
    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar * 3.5;
        }
    "#;
    let client = Client::start();

    let pro_que = ProQue::builder().
            queue_properties(ocl::CommandQueueProperties::new().profiling())
        .src(src)
        .dims(1 << 20)
        .build()?;
    let device = pro_que.device();

    println!("Device: {:?}", device.name().unwrap());

    let buffer = pro_que.create_buffer::<f32>()?;

    let calibration_event = pro_que.queue().enqueue_marker::<ocl::Event>(None)?;
    calibration_event.wait_for()?;
    let calibration_start = get_start_time(&calibration_event)?;
    let calibration_end = get_end_time(&calibration_event)?;

     let gpu_context = client.new_gpu_context(
         Some("MyContext"),
         tracy_client::GpuContextType::OpenCL,
         calibration_end, 
         1.0f32, // OpenCL timestamps are in nanoseconds 
     ).unwrap();
    println!("Calibration Event Start: {}", calibration_start);
    println!("Calibration Event End:   {}", calibration_end);

    //let zone = span!("OCL");
    let mut zone = gpu_context.span_alloc("OCL Dummy","main","main.rs",35).unwrap();
    //zone.emit_color(0xFF0000);

    let kernel = pro_que.kernel_builder("add")
        .arg(&buffer)
        .arg(10.0f32)
        .build()?;

    let mut event = ocl::Event::empty();
    unsafe { kernel.cmd().enew(&mut event).enq()? };

    {
    let _cpu_wait_zone = span!("clWaitForEvents");
    event.wait_for()?;
    }

    zone.end_zone();
    
    let start_time = get_start_time(&event)?;
    println!("Start time: {}", start_time);
    zone.upload_timestamp_start(start_time);
    let end_time = get_end_time(&event)?;
    println!("End time:   {}", end_time);
    zone.upload_timestamp_end(end_time);


    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}
