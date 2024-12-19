
@KERNEL
{
    fpreal3 dP = 0;
    fpreal dPw = 0;
    int nprims = @prims.entries;
    for (int i = 0; i < nprims; ++i)
    {
        int prim = @prims.comp(i);
        int pt0 = @pts.tupleAt(prim)[0];
        int pt1 = @pts.tupleAt(prim)[1];
        int primary = 1;
        if (pt1 == @elemnum)
        {
            pt1 = pt0;
            primary = 0;
        }
        // Using @P(@elemnum) uses fewer registers than @P
        // so use the former to keep register usage to 32
        fpreal3 p0 = @P(@elemnum);
        float invmass0 = invMass(@mass.data, 0, @elemnum);
        fpreal3 p1 = @P(pt1);
        float invmass1 = invMass(@mass.data, 0, pt1);
        float wsum = invmass0 + invmass1;
        if (wsum == 0.0f)
            continue;
        fpreal3 n = p1 - p0;
        fpreal d = length(n);
        if (d < 1e-6f)
            continue;
            
        float restlen = @restlength(prim);
        float kstiff = (d < restlen) ? @compressstiffness(prim) : @stiffness(prim);
        fpreal dL = 0;
        if (kstiff > 1e-6f)
        {
            fpreal l = @L.tuple[prim];
           
            // XPBD term
            fpreal alpha = 1.0f / kstiff;
            alpha /= @TimeInc * @TimeInc;
           
            // Constraint calculation
            fpreal C = d - restlen;
            n /= d;
            fpreal3 gradC = n;
           
            dL = (-C - alpha * l) / (wsum + alpha);
            dP += invmass0 * n * -dL;
            dPw += 1;
        }
        // Only updated dL if this point is first in constraint
        if (primary)
            @dL.tuple[prim] = dL;
    }
    
    // Push out of collision volume
    fpreal offset = @usepscale ? @pscale : @sdfoffset;
    float3 pos = @P(@elemnum) + dP / dPw;
    fpreal dist = @sdf.worldSample(pos) - offset;
    if (dist < 0) {
        fpreal3 dir = @sdf.worldGradient(pos);
        dir = normalize(dir);
        dP -= dir * dist;
        dPw += 1;
    }
    
    // Push out of height field
    // if (@useground > 0)
    // {
    //     pos = @P(@elemnum) + dP / dPw;
    //     fpreal3 up = (fpreal3)(
    //         @height.xformtoworld.s8,
    //         @height.xformtoworld.s9,
    //         @height.xformtoworld.sA
    //     );
    //     up = normalize(up);
    //     fpreal3 vpos = mat4vec3mul(@height.xformtovoxel, pos);
    //     fpreal dy = dot(up, pos);
    //     dist = (dy - @height.sample(vpos)) - offset;
    //     if (dist < 0) {
    //         dP -= up * dist;
    //         dPw += 1;
    //     }
    // }
    
    
    
    
    // Push out of tangent plane heuristic collider
    // if (@usenormals)
    // {
    //     pos = @P + dP/dPw;
    //     float3 delta = pos - @rest;
    //     float proj = dot(delta, @N);
    //     if (proj < -@tolerance)
    //     {
    //         dP += normalize(delta)*(proj + @tolerance);
    //         dPw += 1;
    //     }
    // }
    

    @dP.set(dP);
    @dPw.set(dPw);
}