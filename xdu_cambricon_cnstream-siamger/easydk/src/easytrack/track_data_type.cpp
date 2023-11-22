#include "track_data_type.h"

namespace edk
{

BoundingBox tlwh2xyah(const BoundingBox &tlwh) {
  BoundingBox xyah;
  xyah.x = tlwh.x + tlwh.width / 2;
  xyah.y = tlwh.y + tlwh.height / 2;
  xyah.width = tlwh.width / tlwh.height;
  xyah.height = tlwh.height;
  return xyah;
}

BoundingBox xyah2tlwh(const BoundingBox &xyah) {
  BoundingBox tlwh;
  tlwh.width = xyah.width * xyah.height;
  tlwh.height = xyah.height;
  tlwh.x = xyah.x - tlwh.width / 2;
  tlwh.y = xyah.y - tlwh.height / 2;
  return tlwh;
}

BoundingBox xyah2tlbr(const BoundingBox &xyah){
  BoundingBox tlbr = xyah2tlwh(xyah);
  tlbr.width = tlbr.width + tlbr.x;
  tlbr.height = tlbr.height + tlbr.y;
  return tlbr;
}


BoundingBox tlwh2xywh(const BoundingBox &tlwh) {
  BoundingBox xywh;
  xywh.width = tlwh.width ;
  xywh.height = tlwh.height;
  xywh.x = tlwh.x + tlwh.width / 2;
  xywh.y = tlwh.y + tlwh.height / 2;
  return xywh;
}

BoundingBox xywh2tlwh(const BoundingBox &xywh) {
  BoundingBox tlwh;
  tlwh.width = xywh.width;
  tlwh.height = xywh.height;
  tlwh.x = xywh.x - tlwh.width / 2;
  tlwh.y = xywh.y - tlwh.height / 2;
  return tlwh;
}

BoundingBox tlwh2xyar(const BoundingBox &tlwh) {
  BoundingBox xyar;
  xyar.width = tlwh.width * tlwh.height;
  xyar.height = tlwh.height / tlwh.width;
  xyar.x = tlwh.x + tlwh.width / 2;
  xyar.y = tlwh.y + tlwh.height / 2;
  return xyar;
}

BoundingBox xyar2tlwh(const BoundingBox &xyar) {
  BoundingBox tlwh;
  tlwh.height = std::sqrt(std::max(0.0f, xyar.height * xyar.width));
  tlwh.width = xyar.width / tlwh.height;
  tlwh.x = xyar.x - tlwh.width / 2;
  tlwh.y = xyar.y - tlwh.height / 2;
  return tlwh;
}

}// namespace edk