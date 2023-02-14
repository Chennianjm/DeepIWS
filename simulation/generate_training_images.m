%% PSF of confocal microscopy
nm=1e-9; um=1e-6; mm=1e-3; lambda_exc=640*nm; lambda_emi=685*nm; n=1.518; NA=1.49;
airy_disc=0.61*lambda_exc/NA; f_cutoff=2*NA/lambda_exc;
N=1000; pixel=10/N; [xp,yp]=meshgrid(-N/2*pixel:pixel:(N/2-1)*pixel,-N/2*pixel:pixel:(N/2-1)*pixel); [phi,rho]=cart2pol(xp,yp);
P_solid=heaviside(1-rho).*exp(-rho.^2).*exp(1i*0); P_hollow=heaviside(1-rho).*exp(-rho.^2).*exp(1i*phi);
J(:,:,1)=1/sqrt(2)*ones(N); J(:,:,2)=1i/sqrt(2)*ones(N);
PSFe_solid=Richards_Wolf(xp,yp,P_solid,J,n,NA); PSFe_hollow=Richards_Wolf(xp,yp,P_hollow,J,n,NA); pixel_exc=lambda_exc/(N*NA*pixel);
[x_exc,y_exc]=meshgrid(-N/2*pixel_exc:pixel_exc:(N/2-1)*pixel_exc,-N/2*pixel_exc:pixel_exc:(N/2-1)*pixel_exc);
[fx,fy]=meshgrid(-1/2/pixel_exc:1/N/pixel_exc:1/2/pixel_exc-1/N/pixel_exc,-1/2/pixel_exc:1/N/pixel_exc:1/2/pixel_exc-1/N/pixel_exc); fx=fx/f_cutoff; fy=fy/f_cutoff;
P_emi=heaviside(1-rho); PSF_emi=Richards_Wolf(xp,yp,P_emi,J,n,NA); pixel_emi=lambda_emi/(N*NA*pixel);
[x_emi,y_emi]=meshgrid(-N/2*pixel_emi:pixel_emi:(N/2-1)*pixel_emi,-N/2*pixel_emi:pixel_emi:(N/2-1)*pixel_emi);
r_pinhole=0.89*airy_disc; pinhole=heaviside(r_pinhole-sqrt(x_emi.^2+y_emi.^2)); PSF_det=(opt_ifft(opt_fft(PSF_emi).*opt_fft(pinhole)));
N_camera=500; pixel_camera=20*nm;
[x_camera,y_camera]=meshgrid(-N_camera/2*pixel_camera:pixel_camera:(N_camera/2-1)*pixel_camera,-N_camera/2*pixel_camera:pixel_camera:(N_camera/2-1)*pixel_camera);
PSFe_solid_camera=interp2(x_exc,y_exc,PSFe_solid,x_camera,y_camera);
PSFe_hollow_camera=interp2(x_exc,y_exc,PSFe_hollow,x_camera,y_camera);
PSF_det_camera=interp2(x_emi,y_emi,PSF_det,x_camera,y_camera);
PSF_solid_camera=PSFe_solid_camera.*PSF_det_camera;
PSF_hollow_camera=PSFe_hollow_camera.*PSF_det_camera;
SR=2; PSF_super_camera=interp2(x_camera/SR,y_camera/SR,PSF_solid_camera,x_camera,y_camera,'linear',0);
OTF_solid=opt_fft(PSF_solid_camera); OTF_hollow=opt_fft(PSF_hollow_camera); OTF_super=opt_fft(PSF_super_camera);

%% degraded image
initial_frame=0; frames=1; sample_mode='bead';
for frame=initial_frame:initial_frame+frames-1
    if strcmp(sample_mode,'bead')
        density=1000*nm;
        sample_binary=zeros(N_camera);threshold=2*rand*(pixel_camera/density).^2;
        sample_binary(rand(N_camera)<threshold)=1;
        sample=sample_binary.*(0.5+0.5*rand(N_camera));
        photon_solid=300+200*rand; 
    elseif strcmp(sample_mode,'tube')
        num_tube=100+randi(100); num_step=400+randi(200);
        sample=MicrotubuleGenerator(N_camera,N_camera,num_tube,num_step);
        photon_solid=100+200*rand;
    end
    image_solid=abs(opt_ifft(opt_fft(sample).*OTF_solid));
    image_hollow=abs(opt_ifft(opt_fft(sample).*OTF_hollow));
    image_super=abs(opt_ifft(opt_fft(sample).*OTF_super));
    Max=max(image_solid(:)); Max_hollow=max(image_hollow(:));
    photon_hollow=Max_hollow/Max*photon_solid;
    image_solid=Noise(image_solid,photon_solid,0.018);
    image_hollow=Noise(image_hollow,photon_hollow,0.018);
    data=zeros(N_camera,N_camera,3);
    data(:,:,1)=image_solid/max(image_solid(:));
    data(:,:,2)=image_hollow/max(image_hollow(:));
    data(:,:,3)=image_super/max(image_super(:));
    data=uint16((2^16-1)*data);
end
    
function image_noise=Noise(image,Max_photon,var_gauss)
    image=double(image)/max(image(:));
    image=uint16(Max_photon*image);
    image_Poisson=imnoise(image,'poisson');
    var_gauss=var_gauss/(2^16-1)^2;
    image_Gauss=imnoise(image_Poisson,'gaussian',0,var_gauss);
    image_Gauss(image_Gauss<0)=0;
    image_noise=double(image_Gauss); 
end

function F = opt_fft(I)
    F=fftshift(fft2(ifftshift(I)));
end

function IF = opt_ifft(I)
    IF=ifftshift(ifft2(fftshift(I)));
end

function PSF = Richards_Wolf(xp,yp,P,p,n,NA)
    [phi,rho]=cart2pol(xp,yp); theta=asin(NA/n*rho);
    Ewx=sqrt(cos(theta)).*P.*((1+(cos(theta)-1).*(cos(phi).^2)).*p(:,:,1)+((cos(theta)-1).*cos(phi).*sin(phi)).*p(:,:,2));
    Ewy=sqrt(cos(theta)).*P.*(((cos(theta)-1).*cos(phi).*sin(phi)).*p(:,:,1)+(1+(cos(theta)-1).*(sin(phi).^2)).*p(:,:,2));
    Ewz=sqrt(cos(theta)).*P.*((sin(theta).*cos(phi)).*p(:,:,1)+(sin(theta).*sin(phi)).*p(:,:,2));
    E(:,:,1)=opt_fft(Ewx./cos(theta));
    E(:,:,2)=opt_fft(Ewy./cos(theta));
    E(:,:,3)=opt_fft(Ewz./cos(theta));
    PSF=E(:,:,1).*conj(E(:,:,1))+E(:,:,2).*conj(E(:,:,2))+E(:,:,3).*conj(E(:,:,3));
end

function img = MicrotubuleGenerator(xcube,ycube,sim_num,step_num)
    SR=3;
    xsize=SR*xcube; ysize=SR*ycube;
    l=1; KT = 4.1; A = 1000; zsize=1;
    forces = zeros(1,sim_num);
    [wlcseries] = WLCmicrotubules(forces,KT,A,l,step_num);
    wlc_centers = mean(wlcseries,1);
    normalized_wlc = wlcseries-repmat(wlc_centers,[step_num 1 1]);
    normalized_wlc(:,1,:) = normalized_wlc(:,1,:)-min(min(normalized_wlc(:,1,:)));
    normalized_wlc(:,2,:) = normalized_wlc(:,2,:)-min(min(normalized_wlc(:,2,:)));
    normalized_wlc(:,3,:) = normalized_wlc(:,3,:)-min(min(normalized_wlc(:,3,:)));
    normalized_wlc(:,1,:) = normalized_wlc(:,1,:)/max(max(abs(normalized_wlc(:,1,:))));
    normalized_wlc(:,2,:) = normalized_wlc(:,2,:)/max(max(abs(normalized_wlc(:,2,:))));
    normalized_wlc(:,3,:) = normalized_wlc(:,3,:)/max(max(abs(normalized_wlc(:,3,:))));
    img_original = zeros(xsize,ysize,zsize);
    for i = 1:sim_num
        for j = 1:step_num
            xcart = round(xsize*normalized_wlc(j,1,i)) + 1;
            ycart = round(ysize*normalized_wlc(j,2,i)) + 1;
            zcart = round(zsize*normalized_wlc(j,3,i)) + 1;
            img_original(xcart,ycart,zcart) = 1;
        end
    end
    while true
        flag_x=(SR-1)*rand+1/2; flag_y=(SR-1)*rand+1/2;
        if ~((flag_x > SR/2-3/5) && (flag_x < SR/2+3/5) && (flag_y > SR/2-3/5) && (flag_y < SR/2+3/5))
            break
        end
    end
    flag_x=flag_x*xcube; flag_y=flag_y*xcube;
    img_original = img_original(1:xsize,1:ysize,1:zsize);
    img = sum(img_original,3);
    img=img(flag_x+1-xcube/2:flag_x+xcube/2,flag_y+1-ycube/2:flag_y+ycube/2);
    img = img./max(img(:));
end    

function [wlcseries]=WLCmicrotubules(forces,KT,A,l,steptot)
    fcand=forces; DNAseriestot=[];
    for ff=1:1:size(fcand,2)
        f=fcand(ff);
        probmax=exp(f*l*1/KT);
        DNAt=[]; DNAt(1,:)=[0 0 0];
        inirnd=rand(1,3)*2-1;
        DNAt(2,:)=inirnd./sqrt(sum(inirnd.^2));
        DNAseries=DNAt;
        indx=3;
        for tt=1:1e8
            dirtemp=(2*rand(3,1)-1);
            direction=dirtemp/sqrt(sum(dirtemp.^2));
            costheta=direction(3);
            phi2=2*(1-DNAt(indx-1,:)*direction);
            prob=exp(f*l*costheta/KT-A/2/l*phi2);
            y=rand*probmax;
            if y<prob
                DNAt=cat(1,DNAt,direction');
                DNAseries=cat(1,DNAseries,DNAseries(indx-1,:)+direction');
                indx=indx+1;
            else
                continue
            end          
            if indx>steptot
                break
            end
        end
        DNAseriestot=cat(3,DNAseriestot,DNAseries);       
    end
    wlcseries=DNAseriestot;
end