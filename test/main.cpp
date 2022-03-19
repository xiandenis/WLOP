
#include "WLOP.h"
#include <iostream>
#include "../include/tiny_obj_loader.h"

void read_obj_file (
						std::string const& filename ,
						std::string const& basepath ,
						std::vector<Eigen::Vector3d>& mMeshVertexs) 
{
	tinyobj::attrib_t mAttrib ;
	std::vector<tinyobj::shape_t> mShapes ;
	std::vector<tinyobj::material_t> mMaterials ; 

	std::string warn ;
	std::string err ;

	bool ret = tinyobj::LoadObj(&mAttrib, &mShapes, &mMaterials, &warn, &err, filename.c_str(),basepath.c_str(), true) ;

	if (!warn.empty()) {
		std::cout <<  "WARN: " << warn << std::endl ;
	}
	if (!err.empty()) {
		std::cout <<  "WARERR: " << err << std::endl ;
	}
	

	int n_vertices = (int)mAttrib.vertices.size () / 3 ;
	mMeshVertexs.resize (n_vertices) ;
	for (int i = 0 ; i < n_vertices ; ++i) {
		int index = i * 3 ;
		mMeshVertexs[i] = Eigen::Vector3d (mAttrib.vertices[index + 0] ,mAttrib.vertices[index + 1] ,mAttrib.vertices[index + 2]) ;
	}
	
}
	

int main (int argc, char** argv) {

	// std::string filename(argv[1]);
	// std::string basepath = filename.substr(0,filename.find_last_of('/') + 1) ;
	// std::string savepath(argv[2]);
	std::string filename = "/home/xx/model/8_Couple.obj";
	std::string basepath = "/home/xx/model" ;
	std::string savepath = "/home/xx/model/8_Couple_reslut.obj";;

	std::vector<Eigen::Vector3d> mMeshVertexs ;
	std::cout << filename << std::endl;
	read_obj_file (filename ,basepath ,mMeshVertexs) ;
	std::cout << mMeshVertexs.size() << std::endl;

	DDX::WLOP wlop (mMeshVertexs ,0.09618 ,80 ,0.45 ,true);
	wlop.Run(50);
	wlop.write_obj (savepath);
	const auto i11x = 10 >> 1;
	std::cout << i11x << std::endl;
	return 0;
}