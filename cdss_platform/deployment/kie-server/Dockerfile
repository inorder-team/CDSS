ARG KIE_VERSION=7.74.1.Final

FROM maven:3.9-eclipse-temurin-17 AS rules_builder
WORKDIR /build/rules/kjar

COPY rules/kjar/pom.xml ./pom.xml
COPY rules/kjar/src ./src
COPY rules/drools-rules ../drools-rules

RUN mvn -B -DskipTests install

FROM quay.io/kiegroup/kie-server:${KIE_VERSION}

USER root
COPY --from=rules_builder /root/.m2/repository/com/inorder/clinical/acs-rules/1.0.0 \
    /opt/jboss/.m2/repository/com/inorder/clinical/acs-rules/1.0.0
COPY deployment/kie-server/start-kie-with-rules.sh /opt/jboss/wildfly/bin/start-kie-with-rules.sh
RUN /opt/jboss/wildfly/bin/add-user.sh -a -u kieserver -p 'kieserver1!' -g kie-server,user --silent \
    && chown -R jboss:root /opt/jboss/.m2/repository/com/inorder/clinical/acs-rules \
    && chown jboss:root /opt/jboss/wildfly/bin/start-kie-with-rules.sh \
    && chmod 755 /opt/jboss/wildfly/bin/start-kie-with-rules.sh
USER jboss

CMD ["./start-kie-with-rules.sh"]
