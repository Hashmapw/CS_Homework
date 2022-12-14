package fun.kwok.natserver.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import fun.kwok.natserver.entity.ResultInfo;
import fun.kwok.natserver.utils.MyPasswordEncoder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;
import javax.servlet.http.HttpServletResponse;
import java.io.PrintWriter;

@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
    @Autowired
    private MyPasswordEncoder myPasswordEncoder;

    @Autowired
    private UserDetailsService myCustomUserService;

    @Autowired
    private ObjectMapper objectMapper;


    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.cors();
        http.csrf().disable();
                http
                .authenticationProvider(authenticationProvider())
                .httpBasic()
                .authenticationEntryPoint((request,response,authException) -> {
                    response.setContentType("application/json;charset=utf-8");
                    response.setStatus(HttpServletResponse.SC_OK);
                    PrintWriter out = response.getWriter();
                    out.write(objectMapper.writeValueAsString(new ResultInfo(false,408,"????????????",null)));
                    out.flush();
                    out.close();
                })
                .and()
                .authorizeRequests()
                .antMatchers("/search","/wechat/**","/CheckCode","/operator/login","/login","/").permitAll()
                .and()
                .authorizeRequests()
                .anyRequest().authenticated()
                .and()
                .exceptionHandling()
                .accessDeniedHandler((request,response,ex) -> {
                    response.setContentType("application/json;charset=utf-8");
                    response.setStatus(HttpServletResponse.SC_OK);
                    PrintWriter out = response.getWriter();
                    out.write(objectMapper.writeValueAsString(new ResultInfo(false,403,"????????????",null)));
                    out.flush();
                    out.close();
                })//????????????????????????
                .and().formLogin().disable()
                .logout().disable();

    }
    @Override
    public void configure(WebSecurity web) {
        //?????????header????????????token??????????????????????????????OPTIONS?????????
        web.ignoring().antMatchers(HttpMethod.OPTIONS, "/**");
    }
    @Bean
    @Override
    public AuthenticationManager authenticationManagerBean() throws Exception {
        return super.authenticationManagerBean();
    }
    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authenticationProvider = new DaoAuthenticationProvider();
        //????????????UserDetailsService????????????
        authenticationProvider.setUserDetailsService(myCustomUserService);
        authenticationProvider.setPasswordEncoder(myPasswordEncoder);
        return authenticationProvider;
    }

    @Bean
    public CorsFilter corsFilter()
    {
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        CorsConfiguration config = new CorsConfiguration();
        // ????????????????????????????????????
        config.setAllowCredentials(true);
        // ?????????????????????
        config.addAllowedOriginPattern("*");
        // ????????????????????????
        config.addAllowedHeader("*");
        // ???????????????????????????
        config.addAllowedMethod("*");
        // ???????????????????????????
        source.registerCorsConfiguration("/**", config);
        return new CorsFilter(source);
    }
}
