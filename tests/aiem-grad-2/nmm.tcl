# Draws Nuclear Magnetic Moment vectors on all atoms
# indicated by the *.nmm file.
# The *.nmm file must be associated with a proper *.pdb file.
#VMD  --- start of VMD description block
#Name:
# NMM
#Synopsis:
# Draws vector arrows from *.nmm file
#Version: 
# 1.1
#Uses VMD version:
# 1.5
#Ease of use:
# 1
#Procedures:
# nmm
#Description:
# After loading a *.pdb file, start this program and when prompted
# read in the associate *.nmm file, select the molecule id and
# the scaling factor.
# The indicated vectors will be displayed using "draw".
#See also:
# the VMD user's guide
#Author: 
# Alexander Kim &lt;alex@spawn.scs.uiuc.edu&gt;
#\VMD  --- end of block

proc nmm {} {
	puts -nonewline "Enter the .nmm file to be read: "
	flush stdout
	set filename [gets stdin]
	set data [open $filename r]

	puts -nonewline "Enter the VMD Molecular ID number: "
	flush stdout
	set vmd_pick_mol [gets stdin]

	puts -nonewline "Enter the color you wish to use (red, blue, etc.): "
	flush stdout
	draw color [gets stdin]

	puts -nonewline "Enter the vector scaling factor: "
	flush stdout
	set scale [gets stdin]

	foreach line [split [read $data] \n] {
		if {[lindex $line 0] == "ATOM"} {
			set vmd_pick_atom [expr {[lindex $line 1] - 1}]

			set sel [atomselect $vmd_pick_mol "index $vmd_pick_atom"]
			set coords1 [lindex [$sel get {x y z}] 0]

			set coords3 "[lindex $line 2] [lindex $line 3] [lindex $line 4]"
			set coords3 [vecscale $scale $coords3]
			set coords2 [vecscale 0.6 $coords3]
			set coords2 [vecadd $coords1 $coords2]
			set coords3 [vecadd $coords1 $coords3]
	
			draw cylinder $coords1 $coords2 radius 0.05
			draw cone $coords2 $coords3 radius 0.2

			puts stdout "Atom $vmd_pick_atom on Molecule $vmd_pick_mol done."
			puts stdout "  VECTOR: $coords1 to $coords2 to $coords3"
		} 
	}	
	close $data
	
}


