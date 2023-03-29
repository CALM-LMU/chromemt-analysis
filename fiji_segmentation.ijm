//@File file

open(file);

// Do CLAHE with blocksize 78px = 100nm, all other params default
// example macro for application to stack from https://imagej.net/plugins/clahe
blocksize = 78;
histogram_bins = 256;
maximum_slope = 3;
mask = "*None*";
fast = true;
process_as_composite = true;

getDimensions( width, height, channels, slices, frames );
isComposite = channels > 1;
parameters =
  "blocksize=" + blocksize +
  " histogram=" + histogram_bins +
  " maximum=" + maximum_slope +
  " mask=" + mask;
if ( fast )
  parameters += " fast_(less_accurate)";
if ( isComposite && process_as_composite ) {
  parameters += " process_as_composite";
  channels = 1;
}
  
for ( f=1; f<=frames; f++ ) {
  Stack.setFrame( f );
  for ( s=1; s<=slices; s++ ) {
    Stack.setSlice( s );
    for ( c=1; c<=channels; c++ ) {
      Stack.setChannel( c );
      run( "Enhance Local Contrast (CLAHE)", parameters );
    }
  }
}

// global Li threshold
// remove outliers (first bright then dark, NOTE: inverted LUT)
setAutoThreshold("Li stack");
run("Convert to Mask", "method=Li background=Light create");
run("Remove Outliers...", "radius=2 threshold=50 which=Bright stack");
run("Remove Outliers...", "radius=2 threshold=50 which=Dark stack");

// save, NOTE: name will be MASK_{original filename} 
dir = File.getParent(file);
save(dir + File.separator + getTitle());

run("Close All");